"""
vae_model.py
============
LSTM Variational Autoencoder for anomaly detection.

Unlike the standard LSTM Autoencoder (models.py), the VAE provides TWO
anomaly signals:
  1. Reconstruction error  — same as standard AE
  2. KL divergence         — measures latent space uncertainty

Anomalies score high on BOTH: they reconstruct poorly AND the model
is uncertain about their latent representation.

Architecture:
  Encoder LSTM → mu, log_var (reparameterization trick) → Decoder LSTM

Score formula:
  anomaly_score = normalize(reconstruction_error + beta * kl_divergence)
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
VAE_WINDOW   = 30
VAE_HIDDEN   = 64
VAE_LATENT   = 16       # larger latent than standard AE for richer distribution
VAE_EPOCHS   = 60
VAE_BATCH    = 32
VAE_PATIENCE = 10
VAE_BETA     = 0.5      # KL weight in loss (beta-VAE framework)
VAE_BETA_SCORE = 1.0    # KL weight in anomaly score


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Variational Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

class LSTMVariationalAutoencoder(nn.Module):
    """
    LSTM-based VAE for time series anomaly detection.

    Encoder: LSTM processes sequence → final hidden state → mu, log_var
    Sampling: z = mu + sigma * epsilon  (reparameterization trick)
    Decoder: z → expand to sequence → LSTM → reconstruct input
    """

    def __init__(self, n_features, hidden=64, latent=16, seq_len=30):
        super().__init__()
        self.seq_len   = seq_len
        self.n_features = n_features
        self.hidden    = hidden
        self.latent    = latent

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_lstm  = nn.LSTM(n_features, hidden, num_layers=2,
                                 batch_first=True, dropout=0.1)
        self.enc_mu    = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_linear = nn.Linear(latent, hidden)
        self.dec_lstm  = nn.LSTM(hidden, hidden, num_layers=2,
                                 batch_first=True, dropout=0.1)
        self.dec_output = nn.Linear(hidden, n_features)

    def encode(self, x):
        """x: (batch, seq_len, features) → mu, log_var: (batch, latent)"""
        _, (h, _) = self.enc_lstm(x)                 # h: (2, batch, hidden)
        h_last = h[-1]                                # (batch, hidden)
        mu     = self.enc_mu(h_last)                  # (batch, latent)
        logvar = self.enc_logvar(h_last)              # (batch, latent)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z from N(mu, sigma^2) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """z: (batch, latent) → reconstruction: (batch, seq_len, features)"""
        h = torch.relu(self.dec_linear(z))            # (batch, hidden)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, hidden)
        d, _ = self.dec_lstm(h)                        # (batch, seq_len, hidden)
        return self.dec_output(d)                      # (batch, seq_len, features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=VAE_BETA):
    """
    VAE loss = Reconstruction Loss + beta * KL Divergence

    - Reconstruction: MSE between input and output
    - KL: Regularizes latent space to be close to N(0, I)
    - beta: Controls the tradeoff (beta-VAE framework)
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_vae(df: pd.DataFrame, name: str) -> tuple:
    """
    Train LSTM-VAE on normal market data (up to TRAIN_END).

    Returns: (model, scaler, thresholds_dict)
    """
    torch.manual_seed(42)
    train_df = df.loc[:TRAIN_END, FEATURE_COLS].dropna()
    log.info(f"[{name}] VAE training rows: {len(train_df):,}")

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(train_df).astype(np.float32)
    X_seq    = make_sequences(X_scaled, VAE_WINDOW)
    log.info(f"[{name}] VAE sequences: {X_seq.shape}")
    n_features = X_seq.shape[2]

    # Train / validation split (90/10)
    X_tensor = torch.tensor(X_seq)
    n_val    = max(1, int(len(X_tensor) * 0.1))
    X_train  = X_tensor[:-n_val]
    X_val    = X_tensor[-n_val:].to(DEVICE)

    loader = DataLoader(TensorDataset(X_train, X_train),
                        batch_size=VAE_BATCH, shuffle=True)

    # Init model
    model     = LSTMVariationalAutoencoder(
        n_features, VAE_HIDDEN, VAE_LATENT, VAE_WINDOW
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop with early stopping
    best_val, patience_count, best_state = float("inf"), 0, None

    for epoch in range(1, VAE_EPOCHS + 1):
        model.train()
        train_loss_sum = 0
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            recon, mu, logvar = model(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_recon, val_mu, val_logvar = model(X_val)
            val_total, val_recon_l, val_kl_l = vae_loss(val_recon, X_val, val_mu, val_logvar)
            val_loss = val_total.item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, patience_count = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if epoch % 10 == 0:
            log.info(
                f"[{name}] VAE epoch {epoch:3d}  "
                f"val_loss={val_loss:.6f}  "
                f"recon={val_recon_l.item():.6f}  "
                f"kl={val_kl_l.item():.6f}"
            )

        if patience_count >= VAE_PATIENCE:
            log.info(f"[{name}] VAE early stop at epoch {epoch}")
            break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Compute thresholds on training data
    model.eval()
    with torch.no_grad():
        X_all = X_tensor.to(DEVICE)
        recon, mu, logvar = model(X_all)
        recon_np = recon.cpu().numpy()
        mu_np    = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()

    # Per-sequence reconstruction error
    mse_per_seq = np.mean((X_seq - recon_np) ** 2, axis=(1, 2))
    # Per-sequence KL divergence
    kl_per_seq  = -0.5 * np.mean(
        1 + logvar_np - mu_np ** 2 - np.exp(logvar_np), axis=1
    )

    thresholds = {
        "recon_mean": float(np.mean(mse_per_seq)),
        "recon_std":  float(np.std(mse_per_seq)),
        "recon_thresh": float(np.mean(mse_per_seq) + 2 * np.std(mse_per_seq)),
        "kl_mean":    float(np.mean(kl_per_seq)),
        "kl_std":     float(np.std(kl_per_seq)),
        "kl_thresh":  float(np.mean(kl_per_seq) + 2 * np.std(kl_per_seq)),
    }

    log.info(
        f"[{name}] VAE thresholds: "
        f"recon={thresholds['recon_thresh']:.6f}  "
        f"kl={thresholds['kl_thresh']:.6f}"
    )

    # Save model and metadata
    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), asset_dir / "vae_model.pt")
    with open(asset_dir / "vae_meta.pkl", "wb") as f:
        pickle.dump({
            "scaler":      scaler,
            "thresholds":  thresholds,
            "window":      VAE_WINDOW,
            "n_features":  n_features,
            "hidden":      VAE_HIDDEN,
            "latent":      VAE_LATENT,
        }, f)
    log.info(f"[{name}] VAE saved → vae_model.pt")
    return model, scaler, thresholds


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def vae_anomaly_score(
    df: pd.DataFrame, model, scaler, thresholds: dict,
    beta_score: float = VAE_BETA_SCORE,
) -> pd.Series:
    """
    Score each day with the trained VAE.

    The anomaly score combines:
      - Normalized reconstruction error (how poorly it reconstructs)
      - Normalized KL divergence (how uncertain the latent encoding is)

    Formula: score = normalize(recon_norm + beta_score * kl_norm)
    """
    X_scaled = scaler.transform(df[FEATURE_COLS].fillna(0)).astype(np.float32)
    X_seq    = make_sequences(X_scaled, VAE_WINDOW)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_seq).to(DEVICE)
        recon, mu, logvar = model(X_tensor)
        recon_np  = recon.cpu().numpy()
        mu_np     = mu.cpu().numpy()
        logvar_np = logvar.cpu().numpy()

    # Per-sequence reconstruction error
    mse_per_seq = np.mean((X_seq - recon_np) ** 2, axis=(1, 2))

    # Per-sequence KL divergence
    kl_per_seq = -0.5 * np.mean(
        1 + logvar_np - mu_np ** 2 - np.exp(logvar_np), axis=1
    )

    # Normalize using training thresholds
    recon_norm = mse_per_seq / (thresholds["recon_thresh"] * 3 + 1e-9)
    kl_norm    = kl_per_seq  / (thresholds["kl_thresh"] * 3 + 1e-9)

    # Combined score
    combined = recon_norm + beta_score * kl_norm

    # Pad the beginning (no windows available for first VAE_WINDOW-1 days)
    full_score = np.concatenate([np.zeros(VAE_WINDOW - 1), combined])

    return pd.Series(full_score.clip(0, 1), index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# Load Helper
# ══════════════════════════════════════════════════════════════════════════════

def load_vae(name: str) -> tuple:
    """Load a saved VAE model from disk."""
    asset_dir = MODELS_DIR / name
    with open(asset_dir / "vae_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    model = LSTMVariationalAutoencoder(
        meta["n_features"], meta["hidden"], meta["latent"], meta["window"]
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(asset_dir / "vae_model.pt", map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, meta["scaler"], meta["thresholds"]
