"""
advanced_models.py
==================
Three advanced models that work alongside the existing 4 baseline models.

Phase 2 Addition — Advanced Model Tier:
  Model A: XGBoost Classifier  — supervised, trained on crash labels
  Model B: HMM Regime Detector — unsupervised, detects bull/bear/crisis states
  Model C: TCN Anomaly Model   — PyTorch Temporal Convolutional Network

Strategy: Keep models.py unchanged (baseline). These advanced models add
capabilities without breaking existing functionality.
"""

import logging
import pickle
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / 'backend' / 'models'
DATA_DIR   = ROOT_DIR / 'backend' / 'data'

# ── Feature columns (gracefully handle missing macro features) ─────────────────
FEATURE_COLS = [
    'log_return', 'zscore_10', 'zscore_20', 'zscore_60',
    'vol_10', 'vol_30', 'vol_ratio', 'drawdown', 'bubble_score',
    'rsi_14', 'bb_position', 'macd_hist', 'volume_zscore',
    'vwap_deviation', 'atr_ratio',
    # macro features (Phase 1 — gracefully optional)
    'yield_curve', 'yield_curve_change', 'dxy_return',
    'dxy_zscore', 'credit_stress', 'vix_level', 'vix_change_5d',
]

TRAIN_END = '2019-12-31'
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Build Supervised Crash Labels
# ═══════════════════════════════════════════════════════════════════════════════

def build_crash_labels(df: pd.DataFrame, asset_name: str, window: int = 5) -> pd.Series:
    """
    Returns a binary Series (0/1) for each row in df.
    1 = within ±window trading days of a known crash event for this asset.
    Uses crash_labels.json from data_loader.
    """
    labels_path = DATA_DIR / 'crash_labels.json'
    with open(labels_path) as f:
        crash_data = json.load(f)

    crash_dates = []
    for event in crash_data['events']:
        if asset_name in event.get('assets_affected', []):
            crash_dates.append(pd.to_datetime(event['date']))

    y = pd.Series(0, index=df.index, dtype=int)
    for cd in crash_dates:
        mask = (df.index >= cd - pd.Timedelta(days=window * 2)) & \
               (df.index <= cd + pd.Timedelta(days=window * 2))
        y[mask] = 1

    pos = y.sum()
    log.info(
        f'[{asset_name}] Crash labels: {pos} positive ({pos/len(y)*100:.1f}%)'
    )
    return y


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A: XGBoost Classifier (Supervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgboost(df: pd.DataFrame, name: str):
    """
    Train XGBoost binary classifier to predict crash events.
    Uses crash_labels.json to build supervised targets.
    """
    # Use only training period to prevent leakage
    train_df = df.loc[:TRAIN_END].copy()

    # Get available feature columns (macro features may not exist yet)
    available_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X = train_df[available_cols].fillna(0).values
    y = build_crash_labels(train_df, name).values

    # Class weight to handle imbalanced labels (crashes are rare)
    pos_weight = max(1, (y == 0).sum() / max((y == 1).sum(), 1))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,  # handles class imbalance
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        tree_method='hist',
    )
    model.fit(X_scaled, y)

    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True, parents=True)
    with open(asset_dir / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler,
                     'feature_cols': available_cols}, f)
    log.info(f'[{name}] XGBoost saved. Features used: {len(available_cols)}')
    return model, scaler, available_cols


def xgboost_score(df: pd.DataFrame, model, scaler, feature_cols) -> pd.Series:
    """Score data with trained XGBoost model. Returns crash probability."""
    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    # Use crash probability (class 1) as anomaly score
    proba = model.predict_proba(X_scaled)[:, 1]
    return pd.Series(proba, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B: HMM Market Regime Detector (Unsupervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_hmm_regime(df: pd.DataFrame, name: str, n_regimes: int = 3):
    """
    Train Hidden Markov Model to detect market regimes.
    3 states: bull (low vol), bear (medium vol), crisis (high vol).
    """
    train_df = df.loc[:TRAIN_END].dropna(subset=['log_return', 'vol_10'])

    # Train on returns + volatility — enough for regime separation
    X = train_df[['log_return', 'vol_10']].values

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type='full',
        n_iter=200,
        random_state=42,
    )
    model.fit(X)

    # Identify which state = crisis (highest volatility state)
    means_vol = [model.means_[i][1] for i in range(n_regimes)]
    crisis_state = int(np.argmax(means_vol))
    bull_state   = int(np.argmin(means_vol))
    bear_state   = [i for i in range(n_regimes)
                    if i != crisis_state and i != bull_state][0]

    regime_map = {crisis_state: 'crisis', bull_state: 'bull', bear_state: 'bear'}

    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True, parents=True)
    with open(asset_dir / 'hmm_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'regime_map': regime_map,
                     'crisis_state': crisis_state}, f)
    log.info(f'[{name}] HMM saved. Crisis={crisis_state}, Bull={bull_state}, Bear={bear_state}')
    return model, regime_map


def hmm_anomaly_score(df: pd.DataFrame, model, regime_map) -> pd.Series:
    """Get probability of being in crisis regime."""
    X = df[['log_return', 'vol_10']].fillna(0).values
    # Get probability of being in crisis state
    posteriors = model.predict_proba(X)   # shape: (n, n_regimes)
    crisis_state = [k for k, v in regime_map.items() if v == 'crisis'][0]
    crisis_prob = posteriors[:, crisis_state]
    return pd.Series(crisis_prob, index=df.index)


def hmm_regime_series(df: pd.DataFrame, model, regime_map) -> pd.Series:
    """Get most likely regime for each time point."""
    X = df[['log_return', 'vol_10']].fillna(0).values
    states = model.predict(X)
    return pd.Series([regime_map.get(s, 'unknown') for s in states], index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C: TCN Anomaly Detector (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

class CausalConv1d(nn.Module):
    """Causal convolution for temporal modeling."""
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad] if self.pad > 0 else self.conv(x)


class TCNAutoencoder(nn.Module):
    """Temporal Convolutional Network autoencoder for anomaly detection."""
    def __init__(self, n_features, channels=32, kernel=3):
        super().__init__()
        self.encoder = nn.Sequential(
            CausalConv1d(n_features, channels, kernel, dilation=1),
            nn.ReLU(),
            CausalConv1d(channels, channels, kernel, dilation=2),
            nn.ReLU(),
            CausalConv1d(channels, channels // 2, kernel, dilation=4),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels // 2, channels, kernel, padding=kernel//2),
            nn.ReLU(),
            nn.ConvTranspose1d(channels, n_features, kernel, padding=kernel//2),
        )

    def forward(self, x):
        # x: (batch, features, seq_len)
        z = self.encoder(x)
        return self.decoder(z)


def train_tcn(df: pd.DataFrame, name: str, window: int = 30, epochs: int = 40):
    """Train TCN autoencoder on normal market patterns."""
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    train_df = df.loc[:TRAIN_END, available_cols].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_df.values)

    # Build windows: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
    seqs = np.array([X_scaled[i:i+window] for i in range(len(X_scaled)-window)])
    seqs_t = torch.tensor(seqs, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)

    model = TCNAutoencoder(n_features=len(available_cols)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loader = DataLoader(TensorDataset(seqs_t), batch_size=32, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total = 0
        for (xb,) in loader:
            pred = model(xb)
            loss = loss_fn(pred, xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0:
            log.info(f'[{name}] TCN epoch {epoch}/{epochs}  loss={total/len(loader):.4f}')

    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), asset_dir / 'tcn_model.pt')
    with open(asset_dir / 'tcn_meta.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'feature_cols': available_cols,
                     'window': window}, f)
    log.info(f'[{name}] TCN saved.')
    return model, scaler, available_cols


def tcn_anomaly_score(df, model, scaler, feature_cols, window=30):
    """Score data with trained TCN. Returns normalized reconstruction error."""
    available = [c for c in feature_cols if c in df.columns]
    X = scaler.transform(df[available].fillna(0).values)
    scores = np.zeros(len(df))
    model.eval()
    with torch.no_grad():
        for i in range(window, len(X)):
            seq = torch.tensor(X[i-window:i], dtype=torch.float32)
            seq_t = seq.unsqueeze(0).permute(0, 2, 1).to(DEVICE)
            recon = model(seq_t).permute(0, 2, 1).squeeze(0).cpu().numpy()
            scores[i] = np.mean((X[i-window:i] - recon) ** 2)
    s = pd.Series(scores, index=df.index)
    lo, hi = np.percentile(scores[scores > 0], [1, 99])
    return ((s.clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Ensemble (Meta-learner Weighted)
# ═══════════════════════════════════════════════════════════════════════════════

def dynamic_ensemble_score(
    z_score, iforest, lstm, prophet, xgb, hmm, tcn,
    weights=None
) -> pd.Series:
    """
    Combine all 7 model scores with weights.
    Default weights give more power to supervised (XGBoost) and deep (LSTM/TCN).
    Weights are tunable after Phase 2 evaluation.
    """
    if weights is None:
        weights = {
            'zscore':  0.05,   # baseline statistical
            'iforest': 0.10,   # unsupervised ML
            'lstm':    0.20,   # deep sequence
            'prophet': 0.10,   # trend deviation
            'xgb':     0.30,   # supervised — highest weight (uses crash labels)
            'hmm':     0.10,   # regime state
            'tcn':     0.15,   # temporal conv network
        }

    scores = {
        'zscore': z_score, 'iforest': iforest, 'lstm': lstm,
        'prophet': prophet, 'xgb': xgb, 'hmm': hmm, 'tcn': tcn
    }

    # Align all series to same index, fill missing with mean
    idx = z_score.index
    combined = sum(
        weights[k] * scores[k].reindex(idx).fillna(0.5)
        for k in weights
    )
    return (combined * 100).clip(0, 100)
