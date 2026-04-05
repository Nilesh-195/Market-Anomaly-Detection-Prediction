"""
anomaly_transformer.py
======================
Anomaly Transformer for time-series anomaly detection.

Based on the ICLR 2022 paper "Anomaly Transformer: Time Series Anomaly
Detection with Association Discrepancy".

Key Innovation — Association Discrepancy:
  Normal time points have strong associations with adjacent points (prior).
  Anomalies break this temporal pattern — their learned associations (series)
  diverge from what's expected (prior).

  anomaly_score = reconstruction_error × association_discrepancy

Architecture:
  - Anomaly Attention: learns both prior (Gaussian) and series (data-driven)
    temporal associations simultaneously
  - Multi-layer encoder with feed-forward blocks
  - Reconstruction head to produce output

This gives TWO orthogonal signals:
  1. Reconstruction quality (like AE/VAE)
  2. Temporal association breakdown (unique to this model)
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from features import make_sequences

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "log_return",
    "zscore_10", "zscore_20", "zscore_60",
    "vol_10",    "vol_30",    "vol_ratio",
    "drawdown",  "bubble_score",
    "rsi_14",    "bb_position", "macd_hist",
    "volume_zscore", "vwap_deviation",
    "atr_ratio",
]

TRAIN_END = "2019-12-31"
DEVICE    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ── Hyperparameters ───────────────────────────────────────────────────────────
AT_WINDOW    = 30
AT_D_MODEL   = 64       # model dimension
AT_N_HEADS   = 4        # attention heads
AT_N_LAYERS  = 2        # encoder layers
AT_D_FF      = 128      # feed-forward dimension
AT_DROPOUT   = 0.1
AT_EPOCHS    = 60
AT_BATCH     = 32
AT_PATIENCE  = 10
AT_LAMBDA    = 1.0      # association discrepancy weight in anomaly score


# ══════════════════════════════════════════════════════════════════════════════
# Anomaly Attention Layer
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyAttention(nn.Module):
    """
    Core of the Anomaly Transformer.

    Computes two types of temporal associations:
      1. Prior Association: Gaussian kernel based on time distance
         P(i,j) ∝ exp(-|i-j|^2 / (2σ^2))
         → What we EXPECT the attention to look like (local focus)

      2. Series Association: Learned attention from Q, K, V
         S(i,j) = softmax(Q·K^T / √d)
         → What the DATA actually shows

    Association Discrepancy = KL(S || P) + KL(P || S)
      → High discrepancy = anomaly (temporal pattern is broken)
    """

    def __init__(self, d_model, n_heads, seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.seq_len = seq_len

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable prior scale (sigma) — one per head
        self.sigma = nn.Parameter(torch.ones(n_heads) * 5.0)

        self.dropout = nn.Dropout(dropout)

    def _prior_association(self, sigma):
        """
        Gaussian prior: P(i,j) ∝ exp(-|i-j|^2 / (2σ^2))
        Returns normalized probability distribution over time.
        """
        positions = torch.arange(self.seq_len, device=sigma.device).float()
        # Distance matrix: |i - j|
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        # Gaussian kernel per head
        # sigma: (n_heads,) → (n_heads, 1, 1)
        sig = sigma.abs().clamp(min=0.5).view(-1, 1, 1)
        prior = torch.exp(-dist.unsqueeze(0) ** 2 / (2 * sig ** 2))
        # Normalize each row to sum to 1
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-9)
        return prior  # (n_heads, seq_len, seq_len)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: output, series_assoc, prior_assoc
        """
        B, L, D = x.shape
        H = self.n_heads

        # Project to Q, K, V
        Q = self.W_q(x).view(B, L, H, self.d_k).transpose(1, 2)  # (B, H, L, d_k)
        K = self.W_k(x).view(B, L, H, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, H, self.d_k).transpose(1, 2)

        # Series Association (learned from data)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        series_assoc = F.softmax(scores, dim=-1)  # (B, H, L, L)
        series_assoc = self.dropout(series_assoc)

        # Prior Association (Gaussian kernel)
        prior_assoc = self._prior_association(self.sigma)  # (H, L, L)
        prior_assoc = prior_assoc.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, L, L)

        # Attention output using series association
        context = torch.matmul(series_assoc, V)  # (B, H, L, d_k)
        context = context.transpose(1, 2).contiguous().view(B, L, D)
        output  = self.W_o(context)

        return output, series_assoc, prior_assoc


# ══════════════════════════════════════════════════════════════════════════════
# Encoder Layer
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyTransformerLayer(nn.Module):
    """Single encoder layer with Anomaly Attention + Feed-Forward."""

    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout=0.1):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads, seq_len, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.ff        = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Anomaly Attention with residual
        attn_out, series, prior = self.attention(x)
        x = self.norm1(x + attn_out)
        # Feed-Forward with residual
        x = self.norm2(x + self.ff(x))
        return x, series, prior


# ══════════════════════════════════════════════════════════════════════════════
# Full Anomaly Transformer
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyTransformerModel(nn.Module):
    """
    Full Anomaly Transformer model.

    Input:  (batch, seq_len, n_features)
    Output: (batch, seq_len, n_features), series_list, prior_list
    """

    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=2,
                 d_ff=128, seq_len=30, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.d_model    = d_model
        self.seq_len    = seq_len

        # Input projection
        self.input_proj  = nn.Linear(n_features, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Encoder layers
        self.layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, d_ff, seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Output projection (reconstruction)
        self.output_proj = nn.Linear(d_model, n_features)

    def forward(self, x):
        """
        x: (batch, seq_len, n_features)
        Returns:
          - recon: (batch, seq_len, n_features)
          - series_list: list of (batch, heads, seq, seq) per layer
          - prior_list:  list of (batch, heads, seq, seq) per layer
        """
        # Project input to model dimension + add positional encoding
        h = self.input_proj(x) + self.pos_encoding

        series_list = []
        prior_list  = []

        for layer in self.layers:
            h, series, prior = layer(h)
            series_list.append(series)
            prior_list.append(prior)

        # Reconstruct
        recon = self.output_proj(h)
        return recon, series_list, prior_list


# ══════════════════════════════════════════════════════════════════════════════
# Association Discrepancy
# ══════════════════════════════════════════════════════════════════════════════

def association_discrepancy(series_list, prior_list):
    """
    Compute Association Discrepancy across all layers.

    AssDis = mean over layers of:
      KL(Series || Prior) + KL(Prior || Series)

    Returns per-sample discrepancy: (batch,)
    """
    total_disc = 0.0
    n_layers   = len(series_list)

    for series, prior in zip(series_list, prior_list):
        # series, prior: (batch, heads, seq, seq)
        # Clamp to avoid log(0)
        s = series.clamp(min=1e-8)
        p = prior.clamp(min=1e-8)

        # Symmetric KL divergence
        kl_sp = (s * (s.log() - p.log())).sum(dim=-1).mean(dim=(1, 2))  # (batch,)
        kl_ps = (p * (p.log() - s.log())).sum(dim=-1).mean(dim=(1, 2))  # (batch,)
        total_disc += (kl_sp + kl_ps)

    return total_disc / n_layers  # (batch,)


def anomaly_transformer_loss(recon, x, series_list, prior_list, lambda_ad=1.0):
    """
    Combined loss:
      L = reconstruction_loss + lambda * association_discrepancy

    The reconstruction loss teaches the model to reconstruct normal patterns.
    The association discrepancy regularizes attention to follow Gaussian prior
    on normal data — so anomalies will MAXIMIZE discrepancy at inference.
    """
    recon_loss = F.mse_loss(recon, x)

    # For training, we MINIMIZE discrepancy (normal data should match prior)
    ad = association_discrepancy(series_list, prior_list).mean()

    total = recon_loss + lambda_ad * ad
    return total, recon_loss, ad


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_anomaly_transformer(df: pd.DataFrame, name: str) -> tuple:
    """
    Train Anomaly Transformer on normal market data (up to TRAIN_END).

    Returns: (model, scaler, thresholds_dict)
    """
    torch.manual_seed(42)
    train_df = df.loc[:TRAIN_END, FEATURE_COLS].dropna()
    log.info(f"[{name}] AT training rows: {len(train_df):,}")

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(train_df).astype(np.float32)
    X_seq    = make_sequences(X_scaled, AT_WINDOW)
    log.info(f"[{name}] AT sequences: {X_seq.shape}")
    n_features = X_seq.shape[2]

    # Train / validation split (90/10)
    X_tensor = torch.tensor(X_seq)
    n_val    = max(1, int(len(X_tensor) * 0.1))
    X_train  = X_tensor[:-n_val]
    X_val    = X_tensor[-n_val:].to(DEVICE)

    loader = DataLoader(TensorDataset(X_train),
                        batch_size=AT_BATCH, shuffle=True)

    # Init model
    model = AnomalyTransformerModel(
        n_features=n_features,
        d_model=AT_D_MODEL,
        n_heads=AT_N_HEADS,
        n_layers=AT_N_LAYERS,
        d_ff=AT_D_FF,
        seq_len=AT_WINDOW,
        dropout=AT_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val, patience_count, best_state = float("inf"), 0, None

    for epoch in range(1, AT_EPOCHS + 1):
        model.train()
        train_loss_sum = 0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            recon, series_list, prior_list = model(xb)
            loss, _, _ = anomaly_transformer_loss(
                recon, xb, series_list, prior_list
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_recon, val_series, val_prior = model(X_val)
            val_total, val_recon_l, val_ad = anomaly_transformer_loss(
                val_recon, X_val, val_series, val_prior
            )
            val_loss = val_total.item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience_count = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if epoch % 10 == 0:
            log.info(
                f"[{name}] AT epoch {epoch:3d}  "
                f"val_loss={val_loss:.6f}  "
                f"recon={val_recon_l.item():.6f}  "
                f"assoc_disc={val_ad.item():.6f}"
            )

        if patience_count >= AT_PATIENCE:
            log.info(f"[{name}] AT early stop at epoch {epoch}")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Compute thresholds on training data
    model.eval()
    with torch.no_grad():
        X_all = X_tensor.to(DEVICE)
        recon, series_list, prior_list = model(X_all)
        recon_np = recon.cpu().numpy()

    # Per-sequence reconstruction error
    mse_per_seq = np.mean((X_seq - recon_np) ** 2, axis=(1, 2))

    # Per-sequence association discrepancy
    with torch.no_grad():
        ad_per_seq = association_discrepancy(series_list, prior_list).cpu().numpy()

    thresholds = {
        "recon_mean":   float(np.mean(mse_per_seq)),
        "recon_std":    float(np.std(mse_per_seq)),
        "recon_thresh": float(np.mean(mse_per_seq) + 2 * np.std(mse_per_seq)),
        "ad_mean":      float(np.mean(ad_per_seq)),
        "ad_std":       float(np.std(ad_per_seq)),
        "ad_thresh":    float(np.mean(ad_per_seq) + 2 * np.std(ad_per_seq)),
    }

    log.info(
        f"[{name}] AT thresholds: "
        f"recon={thresholds['recon_thresh']:.6f}  "
        f"assoc_disc={thresholds['ad_thresh']:.6f}"
    )

    # Save model and metadata
    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), asset_dir / "anomaly_transformer.pt")
    with open(asset_dir / "at_meta.pkl", "wb") as f:
        pickle.dump({
            "scaler":      scaler,
            "thresholds":  thresholds,
            "window":      AT_WINDOW,
            "n_features":  n_features,
            "d_model":     AT_D_MODEL,
            "n_heads":     AT_N_HEADS,
            "n_layers":    AT_N_LAYERS,
            "d_ff":        AT_D_FF,
        }, f)
    log.info(f"[{name}] Anomaly Transformer saved → anomaly_transformer.pt")
    return model, scaler, thresholds


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def at_anomaly_score(
    df: pd.DataFrame, model, scaler, thresholds: dict,
    lambda_score: float = AT_LAMBDA,
) -> pd.Series:
    """
    Score each day with the trained Anomaly Transformer.

    The anomaly score combines:
      - Normalized reconstruction error
      - Normalized association discrepancy (temporal pattern breakdown)

    Formula: score = normalize(recon_norm × (1 + lambda * ad_norm))
    Using multiplication (not addition) because:
      - High recon + high AD = very anomalous (multiplicative boost)
      - High recon + low AD  = unusual but not structurally broken
      - Low recon  + high AD = attention drift but reconstructs okay
    """
    X_scaled = scaler.transform(df[FEATURE_COLS].fillna(0)).astype(np.float32)
    X_seq    = make_sequences(X_scaled, AT_WINDOW)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq).to(DEVICE)
        recon, series_list, prior_list = model(X_tensor)
        recon_np = recon.cpu().numpy()

        # Association discrepancy per sequence
        ad_scores = association_discrepancy(series_list, prior_list).cpu().numpy()

    # Per-sequence reconstruction error
    mse_per_seq = np.mean((X_seq - recon_np) ** 2, axis=(1, 2))

    # Normalize using training thresholds
    recon_norm = mse_per_seq / (thresholds["recon_thresh"] * 3 + 1e-9)
    ad_norm    = ad_scores   / (thresholds["ad_thresh"] * 3 + 1e-9)

    # Multiplicative combination (anomalies score high on BOTH)
    combined = recon_norm * (1 + lambda_score * ad_norm)

    # Pad the beginning
    full_score = np.concatenate([np.zeros(AT_WINDOW - 1), combined])

    return pd.Series(full_score.clip(0, 1), index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Load Helper
# ══════════════════════════════════════════════════════════════════════════════

def load_anomaly_transformer(name: str) -> tuple:
    """Load a saved Anomaly Transformer from disk."""
    asset_dir = MODELS_DIR / name
    with open(asset_dir / "at_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    model = AnomalyTransformerModel(
        n_features=meta["n_features"],
        d_model=meta["d_model"],
        n_heads=meta["n_heads"],
        n_layers=meta["n_layers"],
        d_ff=meta["d_ff"],
        seq_len=meta["window"],
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(asset_dir / "anomaly_transformer.pt",
                    map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, meta["scaler"], meta["thresholds"]
