"""
models.py
=========
Defines and trains all 4 anomaly detection models:

  Model 1 — Z-Score Baseline        (statistical, no training needed)
  Model 2 — Isolation Forest        (sklearn unsupervised ML)
  Model 3 — LSTM Autoencoder        (PyTorch deep learning)
  Model 4 — Prophet Residual        (Facebook Prophet trend decomposition)

Each model:
  - Is trained per-asset independently
  - Returns a normalised anomaly score in [0, 1]  (0 = normal, 1 = extreme)
  - Is saved to backend/models/<ASSET>/

Note: Uses PyTorch (not TensorFlow) — Python 3.14 compatible.
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Z-Score Baseline
# ══════════════════════════════════════════════════════════════════════════════
def zscore_anomaly_score(df: pd.DataFrame) -> pd.Series:
    z = df["zscore_20"].abs().fillna(0)
    return (z / 6.0).clip(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Isolation Forest
# ══════════════════════════════════════════════════════════════════════════════
def train_isolation_forest(df, name, contamination=0.03):
    train_df = df.loc[:TRAIN_END, FEATURE_COLS].dropna()
    log.info(f"[{name}] IF training rows: {len(train_df):,}")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_df)
    model   = IsolationForest(n_estimators=200, contamination=contamination,
                               max_samples="auto", random_state=42, n_jobs=-1)
    model.fit(X_train)
    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True)
    with open(asset_dir / "isolation_forest.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    log.info(f"[{name}] Isolation Forest saved.")
    return model, scaler


def isolation_forest_score(df, model, scaler):
    X    = scaler.transform(df[FEATURE_COLS].fillna(0))
    raw  = model.decision_function(X)
    flip = -raw
    lo, hi = flip.min(), flip.max()
    if hi == lo:
        return pd.Series(0.0, index=df.index)
    return pd.Series((flip - lo) / (hi - lo), index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — LSTM Autoencoder (PyTorch)
# ══════════════════════════════════════════════════════════════════════════════
LSTM_WINDOW   = 30
LSTM_HIDDEN   = 64
LSTM_LATENT   = 8
LSTM_EPOCHS   = 50
LSTM_BATCH    = 32
LSTM_PATIENCE = 8


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=64, latent=8, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.enc_lstm   = nn.LSTM(n_features, hidden, batch_first=True)
        self.enc_linear = nn.Linear(hidden, latent)
        self.enc_relu   = nn.ReLU()
        self.dec_linear = nn.Linear(latent, hidden)
        self.dec_lstm   = nn.LSTM(hidden, hidden, batch_first=True)
        self.dec_output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_relu(self.enc_linear(h[-1]))
        d = self.dec_linear(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        d, _ = self.dec_lstm(d)
        return self.dec_output(d)


def _make_sequences(X, window):
    return np.array([X[i: i + window] for i in range(len(X) - window + 1)])


def train_lstm_autoencoder(df, name):
    torch.manual_seed(42)
    train_df = df.loc[:TRAIN_END, FEATURE_COLS].dropna()
    log.info(f"[{name}] LSTM training rows: {len(train_df):,}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(train_df).astype(np.float32)
    X_seq    = _make_sequences(X_scaled, LSTM_WINDOW)
    log.info(f"[{name}] LSTM sequences: {X_seq.shape}")
    n_features = X_seq.shape[2]

    X_tensor = torch.tensor(X_seq)
    n_val    = max(1, int(len(X_tensor) * 0.1))
    loader   = DataLoader(TensorDataset(X_tensor[:-n_val], X_tensor[:-n_val]),
                          batch_size=LSTM_BATCH, shuffle=False)

    model     = LSTMAutoencoder(n_features, LSTM_HIDDEN, LSTM_LATENT, LSTM_WINDOW).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val, patience_count, best_state = float("inf"), 0, None
    X_val = X_tensor[-n_val:].to(DEVICE)

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), xb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), X_val).item()
        if val_loss < best_val:
            best_val, patience_count = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
        if epoch % 10 == 0:
            log.info(f"[{name}] LSTM epoch {epoch:3d}  val_loss={val_loss:.6f}")
        if patience_count >= LSTM_PATIENCE:
            log.info(f"[{name}] LSTM early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_pred = model(X_tensor.to(DEVICE)).cpu().numpy()
    mse     = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
    thresh  = float(np.mean(mse) + 2 * np.std(mse))
    log.info(f"[{name}] LSTM threshold: {thresh:.6f}")

    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), asset_dir / "lstm_autoencoder.pt")
    with open(asset_dir / "lstm_meta.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "threshold": thresh, "window": LSTM_WINDOW,
                     "n_features": n_features, "hidden": LSTM_HIDDEN, "latent": LSTM_LATENT}, f)
    log.info(f"[{name}] LSTM Autoencoder saved → lstm_autoencoder.pt")
    return model, scaler, thresh


def lstm_anomaly_score(df, model, scaler, threshold):
    X_scaled = scaler.transform(df[FEATURE_COLS].fillna(0)).astype(np.float32)
    X_seq    = _make_sequences(X_scaled, LSTM_WINDOW)
    model.eval()
    with torch.no_grad():
        X_pred = model(torch.tensor(X_seq).to(DEVICE)).cpu().numpy()
    mse   = np.mean((X_seq - X_pred) ** 2, axis=(1, 2))
    score = np.concatenate([np.zeros(LSTM_WINDOW - 1), mse])
    return pd.Series((score / (threshold * 3 + 1e-9)).clip(0, 1), index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — Prophet Residual Detector
# ══════════════════════════════════════════════════════════════════════════════
def train_prophet(df, name):
    from prophet import Prophet
    train_df   = df.loc[:TRAIN_END, ["Close"]].copy().reset_index()
    train_df.columns = ["ds", "y"]
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    log.info(f"[{name}] Prophet training rows: {len(train_df):,}")

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, changepoint_prior_scale=0.05,
                    interval_width=0.95)
    model.fit(train_df)

    fc           = model.predict(train_df)
    residual_std = float(np.std(train_df["y"].values - fc["yhat"].values))

    asset_dir = MODELS_DIR / name
    asset_dir.mkdir(exist_ok=True)
    with open(asset_dir / "prophet_model.pkl", "wb") as f:
        pickle.dump({"model": model, "residual_std": residual_std}, f)
    log.info(f"[{name}] Prophet residual_std={residual_std:.4f} saved.")
    return model, residual_std


def prophet_anomaly_score(df, model, residual_std):
    future = df[["Close"]].copy().reset_index()
    future.columns = ["ds", "y"]
    future["ds"] = pd.to_datetime(future["ds"])
    fc    = model.predict(future)
    score = (np.abs(future["y"].values - fc["yhat"].values)
             / (3 * residual_std + 1e-9)).clip(0, 1)
    return pd.Series(score, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════
WEIGHTS = {"zscore": 0.15, "iforest": 0.25, "lstm": 0.40, "prophet": 0.20}


def ensemble_score(zscore_s, iforest_s, lstm_s, prophet_s):
    combined = (WEIGHTS["zscore"]  * zscore_s  +
                WEIGHTS["iforest"] * iforest_s +
                WEIGHTS["lstm"]    * lstm_s    +
                WEIGHTS["prophet"] * prophet_s)
    return (combined * 100).clip(0, 100)


def risk_label(score):
    if score < 40: return "Normal"
    if score < 60: return "Elevated"
    if score < 75: return "High Risk"
    return "Extreme Anomaly"


# ── Load helpers ──────────────────────────────────────────────────────────────
def load_isolation_forest(name):
    with open(MODELS_DIR / name / "isolation_forest.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["scaler"]


def load_lstm(name):
    with open(MODELS_DIR / name / "lstm_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    model = LSTMAutoencoder(meta["n_features"], meta["hidden"],
                             meta["latent"], meta["window"]).to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / name / "lstm_autoencoder.pt",
                                      map_location=DEVICE, weights_only=True))
    model.eval()
    return model, meta["scaler"], meta["threshold"]


def load_prophet(name):
    with open(MODELS_DIR / name / "prophet_model.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["residual_std"]
