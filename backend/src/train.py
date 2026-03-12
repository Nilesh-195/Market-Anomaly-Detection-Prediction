"""
train.py
========
Orchestrates training all 4 anomaly detection models for every asset.

Run from project root:
    python backend/src/train.py

Outputs per asset in backend/models/<ASSET>/:
    isolation_forest.pkl
    lstm_autoencoder.pt + lstm_meta.pkl
    prophet_model.pkl
    scores_all.parquet   ← all 4 model scores + ensemble + risk_label
"""

import logging
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "backend" / "models"

# ── local imports ─────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(ROOT_DIR / "backend" / "src"))

from features import load_all_features
from models import (
    FEATURE_COLS,
    zscore_anomaly_score,
    train_isolation_forest, isolation_forest_score,
    train_lstm_autoencoder, lstm_anomaly_score,
    train_prophet,          prophet_anomaly_score,
    ensemble_score,         risk_label,
)

ASSETS = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]


def train_asset(name: str, df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"\n{'═'*60}")
    log.info(f"  Training all models for: {name}")
    log.info(f"{'═'*60}")
    t0 = time.time()

    # ── 1. Z-Score (no training) ──────────────────────────────────────────────
    log.info(f"[{name}] Model 1 — Z-Score (no training)")
    z_scores = zscore_anomaly_score(df)

    # ── 2. Isolation Forest ───────────────────────────────────────────────────
    log.info(f"[{name}] Model 2 — Isolation Forest")
    if_model, if_scaler = train_isolation_forest(df, name)
    if_scores = isolation_forest_score(df, if_model, if_scaler)

    # ── 3. LSTM Autoencoder ───────────────────────────────────────────────────
    log.info(f"[{name}] Model 3 — LSTM Autoencoder (PyTorch)")
    lstm_model, lstm_scaler, lstm_thresh = train_lstm_autoencoder(df, name)
    lstm_scores = lstm_anomaly_score(df, lstm_model, lstm_scaler, lstm_thresh)

    # ── 4. Prophet ────────────────────────────────────────────────────────────
    log.info(f"[{name}] Model 4 — Prophet Residual")
    prophet_model, residual_std = train_prophet(df, name)
    prophet_scores = prophet_anomaly_score(df, prophet_model, residual_std)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    ens = ensemble_score(z_scores, if_scores, lstm_scores, prophet_scores)

    results = pd.DataFrame({
        "zscore_score":   z_scores,
        "iforest_score":  if_scores,
        "lstm_score":     lstm_scores,
        "prophet_score":  prophet_scores,
        "ensemble_score": ens,
        "risk_label":     ens.apply(risk_label),
    }, index=df.index)

    # ── Save scores ───────────────────────────────────────────────────────────
    out_path = MODELS_DIR / name / "scores_all.parquet"
    results.to_parquet(out_path)
    elapsed = time.time() - t0
    log.info(f"[{name}] ✅ Done in {elapsed:.1f}s  →  {out_path}")

    # ── Quick sanity: top anomaly days ────────────────────────────────────────
    top5 = results["ensemble_score"].nlargest(5)
    log.info(f"[{name}] Top 5 anomaly dates:\n{top5.to_string()}")
    return results


def train_all():
    log.info("Loading feature data …")
    all_features = load_all_features()

    summary = {}
    for name in ASSETS:
        if name not in all_features:
            log.warning(f"No features found for {name}, skipping.")
            continue
        df = all_features[name]
        log.info(f"[{name}] rows={len(df):,}  cols={len(df.columns)}")
        results = train_asset(name, df)
        summary[name] = {
            "rows":        len(results),
            "anomalies":   int((results["ensemble_score"] >= 60).sum()),
            "extreme":     int((results["ensemble_score"] >= 75).sum()),
            "max_score":   float(results["ensemble_score"].max()),
            "mean_score":  float(results["ensemble_score"].mean()),
        }

    log.info("\n" + "═"*60)
    log.info("  TRAINING COMPLETE — Summary")
    log.info("═"*60)
    for name, stats in summary.items():
        log.info(
            f"  {name:<8}  rows={stats['rows']:,}  "
            f"anomalies={stats['anomalies']:3d}  "
            f"extreme={stats['extreme']:3d}  "
            f"max={stats['max_score']:.1f}  "
            f"mean={stats['mean_score']:.1f}"
        )


if __name__ == "__main__":
    train_all()
