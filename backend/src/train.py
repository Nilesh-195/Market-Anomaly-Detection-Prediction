"""
train.py
========
Orchestrates training all 7 anomaly detection models for every asset.

Phase 2 Update: Now trains 4 baseline models + 3 advanced models

Run from project root:
    python backend/src/train.py

Outputs per asset in backend/models/<ASSET>/:
    Baseline Models (4):
      - isolation_forest.pkl
      - lstm_autoencoder.pt + lstm_meta.pkl
      - prophet_model.pkl
    Advanced Models (3):
      - xgboost_model.pkl
      - hmm_model.pkl
      - tcn_model.pt + tcn_meta.pkl
    Output:
      - scores_all.parquet   ← all 7 model scores + both ensembles + risk_label
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
from advanced_models import (
    train_xgboost, xgboost_score,
    train_hmm_regime, hmm_anomaly_score, hmm_regime_series,
    train_tcn, tcn_anomaly_score,
    dynamic_ensemble_score,
)
from dl_models import train_lstm_seq2seq, train_transformer
from gb_models import train_xgboost_regressor

import torch
import pickle

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

    # ── Baseline Ensemble (4 models) ──────────────────────────────────────────
    ens = ensemble_score(z_scores, if_scores, lstm_scores, prophet_scores)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Advanced Models
    # ══════════════════════════════════════════════════════════════════════════

    # ── Advanced Model A — XGBoost ────────────────────────────────────────────
    log.info(f"[{name}] Advanced Model A — XGBoost Classifier")
    xgb_model, xgb_scaler, xgb_cols = train_xgboost(df, name)
    xgb_scores = xgboost_score(df, xgb_model, xgb_scaler, xgb_cols)

    # ── Advanced Model B — HMM Regime ─────────────────────────────────────────
    log.info(f"[{name}] Advanced Model B — HMM Regime Detector")
    hmm_model, regime_map = train_hmm_regime(df, name)
    hmm_scores  = hmm_anomaly_score(df, hmm_model, regime_map)
    hmm_regimes = hmm_regime_series(df, hmm_model, regime_map)

    # ── Advanced Model C — TCN ────────────────────────────────────────────────
    log.info(f"[{name}] Advanced Model C — TCN Autoencoder")
    tcn_model, tcn_scaler, tcn_cols = train_tcn(df, name)
    tcn_scores = tcn_anomaly_score(df, tcn_model, tcn_scaler, tcn_cols)

    # ── Dynamic Ensemble (all 7 models) ───────────────────────────────────────
    adv_ens = dynamic_ensemble_score(
        z_scores, if_scores, lstm_scores, prophet_scores,
        xgb_scores, hmm_scores, tcn_scores
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Deep Learning Price Forecasting Models (NEW)
    # ══════════════════════════════════════════════════════════════════════════

    log.info(f"\n[{name}] Training Price Forecasting Models (Phase 3)...")

    # Get close prices for Phase 3 models
    close_prices = df["Close"].values

    # ── DL Model 1: LSTM Seq2Seq ──────────────────────────────────────────────
    # NOTE: LSTM Seq2Seq has tensor dimension complexity; using API endpoints instead
    log.info(f"[{name}] DL Model 1 — LSTM Seq2Seq (using API endpoint /forecast/lstm)")

    # ── DL Model 2: Transformer ───────────────────────────────────────────────
    # NOTE: Transformer has positional encoding complexity; using API endpoint instead
    log.info(f"[{name}] DL Model 2 — Transformer (using API endpoint /forecast/transformer)")

    # ── DL Model 3: XGBoost Price Regressor ───────────────────────────────────
    try:
        log.info(f"[{name}] DL Model 3 — XGBoost Price Regressor")
        # Use Close prices and engineered features (if available)
        xgb_price_models, xgb_price_scaler, xgb_feature_names = train_xgboost_regressor(
            close_prices, df, name, lookback=30, horizon=30, test_size=0.3
        )
        # Save models and metadata
        meta_path = MODELS_DIR / name / "xgboost_meta.pkl"
        meta_dict = {
            "models": xgb_price_models,
            "scaler": xgb_price_scaler,
            "feature_names": xgb_feature_names
        }
        with open(meta_path, 'wb') as f:
            pickle.dump(meta_dict, f)
        log.info(f"[{name}] XGBoost Price Regressor saved to {meta_path}")
    except Exception as e:
        log.error(f"[{name}] XGBoost Price Regressor training failed: {e}")

    results = pd.DataFrame({
        # Baseline models (4)
        "zscore_score":   z_scores,
        "iforest_score":  if_scores,
        "lstm_score":     lstm_scores,
        "prophet_score":  prophet_scores,
        # Advanced models (3)
        "xgb_score":      (xgb_scores * 100).clip(0, 100),
        "hmm_score":      (hmm_scores * 100).clip(0, 100),
        "tcn_score":      (tcn_scores * 100).clip(0, 100),
        "hmm_regime":     hmm_regimes,
        # Ensembles
        "ensemble_score": ens,            # baseline ensemble
        "adv_ensemble":   adv_ens,        # advanced ensemble
        "risk_label":     ens.apply(risk_label),
        "adv_risk":       adv_ens.apply(risk_label),
    }, index=df.index)

    # ── Save scores ───────────────────────────────────────────────────────────
    out_path = MODELS_DIR / name / "scores_all.parquet"
    results.to_parquet(out_path)
    elapsed = time.time() - t0
    log.info(f"[{name}] ✅ Done in {elapsed:.1f}s  →  {out_path}")

    # ── Quick sanity: top anomaly days (advanced ensemble) ────────────────────
    top5 = results["adv_ensemble"].nlargest(5)
    log.info(f"[{name}] Top 5 anomaly dates (adv_ensemble):\n{top5.to_string()}")
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
            "rows":          len(results),
            "baseline_anom": int((results["ensemble_score"] >= 60).sum()),
            "baseline_extr": int((results["ensemble_score"] >= 75).sum()),
            "adv_anom":      int((results["adv_ensemble"] >= 60).sum()),
            "adv_extr":      int((results["adv_ensemble"] >= 75).sum()),
            "max_baseline":  float(results["ensemble_score"].max()),
            "max_adv":       float(results["adv_ensemble"].max()),
            "mean_baseline": float(results["ensemble_score"].mean()),
            "mean_adv":      float(results["adv_ensemble"].mean()),
        }

    log.info("\n" + "═"*60)
    log.info("  TRAINING COMPLETE — Summary")
    log.info("═"*60)
    log.info(f"  {'Asset':<8}  {'Rows':<6}  {'Baseline':<20}  {'Advanced':<20}")
    log.info("  " + "─"*58)
    for name, stats in summary.items():
        log.info(
            f"  {name:<8}  {stats['rows']:>5,}  "
            f"anom={stats['baseline_anom']:3d}  max={stats['max_baseline']:5.1f}  "
            f"anom={stats['adv_anom']:3d}  max={stats['max_adv']:5.1f}"
        )


if __name__ == "__main__":
    train_all()
