"""
predict.py
==========
Generates predictions for a given asset:

  1. Current anomaly analysis  — latest ensemble score + per-model breakdown
  2. Advanced anomaly analysis — advanced ensemble + regime + all 7 models (Phase 2)
  3. 5-day anomaly forecast    — ARIMA on ensemble_score series
  4. Historical anomaly events — top anomaly windows from scores_all.parquet
  5. Regime timeline          — HMM market state history (Phase 2)

Used by the FastAPI backend (api/main.py).

Run standalone:
    python backend/src/predict.py SP500
"""

import logging
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "backend" / "models"
DATA_DIR   = ROOT_DIR / "backend" / "data" / "processed"
sys.path.insert(0, str(ROOT_DIR / "backend" / "src"))

from models import (
    load_isolation_forest, load_lstm, load_prophet,
    zscore_anomaly_score, isolation_forest_score,
    lstm_anomaly_score, prophet_anomaly_score,
    ensemble_score, risk_label,
)
from features import ASSETS, load_all_features


# ── 1. Current Analysis ────────────────────────────────────────────────────────
def current_analysis(asset: str) -> dict:
    """Return today's anomaly scores + price/vol/zscore for the asset."""
    scores_path   = MODELS_DIR / asset / "scores_all.parquet"
    features_path = DATA_DIR   / f"{asset}_features.parquet"

    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest
    ens    = float(latest["ensemble_score"])
    ens_prev = float(prev["ensemble_score"])

    result = {
        "asset":          asset,
        "date":           str(df.index[-1].date()),
        "ensemble_score": round(ens, 2),
        "risk_label":     risk_label(ens),
        "score_delta":    round(ens - ens_prev, 2),
        "model_scores": {
            "zscore":  round(float(latest["zscore_score"]) * 100, 2),
            "iforest": round(float(latest["iforest_score"]) * 100, 2),
            "lstm":    round(float(latest["lstm_score"]) * 100, 2),
            "prophet": round(float(latest["prophet_score"]) * 100, 2),
        },
    }

    # Enrich with price and feature data if available
    if features_path.exists():
        feat = pd.read_parquet(features_path)
        feat.index = pd.to_datetime(feat.index)
        # Align to last common date
        common = feat.index.intersection(df.index)
        if len(common):
            last_feat = feat.loc[common[-1]]
            prev_feat = feat.loc[common[-2]] if len(common) >= 2 else last_feat
            price      = float(last_feat["Close"])
            prev_price = float(prev_feat["Close"])
            result["price"]            = round(price, 2)
            result["price_change_pct"] = round((price / prev_price - 1) * 100, 3)
            result["zscore"]           = round(float(last_feat["zscore_20"]), 4)
            result["volatility"]       = round(float(last_feat["vol_30"]), 4)
            result["drawdown"]         = round(float(last_feat["drawdown"]), 4)

    return result


# ── 1B. Advanced Current Analysis (Phase 2) ───────────────────────────────────
def advanced_current_analysis(asset: str) -> dict:
    """
    Return today's advanced model scores + regime state.
    Includes all 7 model scores (4 baseline + 3 advanced) + both ensembles.
    """
    scores_path   = MODELS_DIR / asset / "scores_all.parquet"
    features_path = DATA_DIR   / f"{asset}_features.parquet"

    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest

    # Advanced ensemble
    adv_ens = float(latest.get("adv_ensemble", latest["ensemble_score"]))
    adv_prev = float(prev.get("adv_ensemble", prev["ensemble_score"]))

    result = {
        "asset":              asset,
        "date":               str(df.index[-1].date()),
        "baseline_ensemble":  round(float(latest["ensemble_score"]), 2),
        "advanced_ensemble":  round(adv_ens, 2),
        "score_delta":        round(adv_ens - adv_prev, 2),
        "risk_label":         risk_label(adv_ens),
        "current_regime":     str(latest.get("hmm_regime", "unknown")),
        "model_scores": {
            # Baseline (4)
            "zscore":  round(float(latest["zscore_score"]) * 100, 2),
            "iforest": round(float(latest["iforest_score"]) * 100, 2),
            "lstm":    round(float(latest["lstm_score"]) * 100, 2),
            "prophet": round(float(latest["prophet_score"]) * 100, 2),
            # Advanced (3)
            "xgb":     round(float(latest.get("xgb_score", 0)), 2),
            "hmm":     round(float(latest.get("hmm_score", 0)), 2),
            "tcn":     round(float(latest.get("tcn_score", 0)), 2),
        },
    }

    # Enrich with price and feature data if available
    if features_path.exists():
        feat = pd.read_parquet(features_path)
        feat.index = pd.to_datetime(feat.index)
        common = feat.index.intersection(df.index)
        if len(common):
            last_feat = feat.loc[common[-1]]
            prev_feat = feat.loc[common[-2]] if len(common) >= 2 else last_feat
            price      = float(last_feat["Close"])
            prev_price = float(prev_feat["Close"])
            result["price"]            = round(price, 2)
            result["price_change_pct"] = round((price / prev_price - 1) * 100, 3)
            result["zscore"]           = round(float(last_feat["zscore_20"]), 4)
            result["volatility"]       = round(float(last_feat["vol_30"]), 4)
            result["drawdown"]         = round(float(last_feat["drawdown"]), 4)

    return result


# ── 1C. Regime Timeline (Phase 2) ──────────────────────────────────────────────
def regime_timeline(asset: str) -> dict:
    """
    Return full HMM regime history plus transition matrix and regime statistics.
    """
    scores_path = MODELS_DIR / asset / "scores_all.parquet"
    features_path = DATA_DIR / f"{asset}_features.parquet"

    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)

    if "hmm_regime" not in df.columns:
        return {
            "asset": asset,
            "error": "HMM regime data not available. Retrain models with Phase 2."
        }

    # Regime timeline
    timeline = []
    for date, row in df.iterrows():
        timeline.append({
            "date":   str(date.date()),
            "regime": row["hmm_regime"],
            "score":  round(float(row.get("adv_ensemble", row["ensemble_score"])), 2),
        })

    # Calculate regime statistics
    regimes = df["hmm_regime"].value_counts().to_dict()
    total = len(df)
    regime_stats = {k: {"count": int(v), "pct": round(v/total*100, 1)}
                    for k, v in regimes.items()}

    # Transition matrix (approximate)
    transitions = {"bull->bear": 0, "bull->crisis": 0, "bear->bull": 0,
                   "bear->crisis": 0, "crisis->bull": 0, "crisis->bear": 0}
    for i in range(1, len(df)):
        prev_r = df["hmm_regime"].iloc[i-1]
        curr_r = df["hmm_regime"].iloc[i]
        if prev_r != curr_r:
            key = f"{prev_r}->{curr_r}"
            if key in transitions:
                transitions[key] += 1

    # Average returns per regime
    avg_returns = {}
    if features_path.exists():
        feat = pd.read_parquet(features_path)
        feat.index = pd.to_datetime(feat.index)
        merged = df.join(feat[["log_return"]], how="left")
        for regime in ["bull", "bear", "crisis"]:
            mask = merged["hmm_regime"] == regime
            if mask.sum() > 0:
                avg_ret = merged.loc[mask, "log_return"].mean() * 100  # to %
                avg_returns[regime] = round(float(avg_ret), 4)

    return {
        "asset":          asset,
        "timeline":       timeline,
        "regime_stats":   regime_stats,
        "transitions":    transitions,
        "avg_returns":    avg_returns,
        "current_regime": timeline[-1]["regime"] if timeline else "unknown",
    }


# ── 2. 5-Day Forecast ──────────────────────────────────────────────────────────
def forecast_anomaly(asset: str, days: int = 5) -> dict:
    """ARIMA forecast of ensemble_score for next `days` trading days."""
    from statsmodels.tsa.arima.model import ARIMA

    scores_path = MODELS_DIR / asset / "scores_all.parquet"
    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)
    series = df["ensemble_score"].dropna()

    # Fit ARIMA(2,1,2) on last 252 trading days (1 year)
    window = series.iloc[-252:]
    try:
        model  = ARIMA(window, order=(2, 1, 2))
        fitted = model.fit()
        fc     = fitted.forecast(steps=days)
        conf   = fitted.get_forecast(steps=days).conf_int(alpha=0.2)
    except Exception as e:
        log.warning(f"[{asset}] ARIMA failed ({e}) — using naive forecast")
        last   = float(series.iloc[-1])
        fc     = pd.Series([last] * days)
        conf   = pd.DataFrame({"lower ensemble_score": [last] * days,
                               "upper ensemble_score": [last] * days})

    last_date = df.index[-1]
    dates = []
    d = last_date
    for _ in range(days):
        d += timedelta(days=1)
        while d.weekday() >= 5:          # skip weekends
            d += timedelta(days=1)
        dates.append(str(d.date()))

    forecast_points = []
    for i in range(days):
        score   = float(np.clip(fc.iloc[i], 0, 100))
        lo      = float(np.clip(conf.iloc[i, 0], 0, 100))
        hi      = float(np.clip(conf.iloc[i, 1], 0, 100))
        forecast_points.append({
            "date":          dates[i],
            "score":         round(score, 2),
            "lower":         round(lo, 2),
            "upper":         round(hi, 2),
            "risk_label":    risk_label(score),
        })

    return {
        "asset":     asset,
        "generated": str(datetime.now().date()),
        "horizon":   days,
        "forecast":  forecast_points,
    }


# ── 3. Historical Anomaly Events ───────────────────────────────────────────────
def historical_anomalies(asset: str, top_n: int = 20,
                          threshold: float = 60.0) -> dict:
    """Return top anomaly events + full chart_data series for the asset."""
    scores_path   = MODELS_DIR / asset / "scores_all.parquet"
    features_path = DATA_DIR   / f"{asset}_features.parquet"

    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)

    flagged = df[df["ensemble_score"] >= threshold]["ensemble_score"].sort_values(ascending=False)

    # Cluster: keep only 1 entry per 5-day window
    events, last_date = [], None
    for date, score in flagged.items():
        if last_date is None or abs((pd.Timestamp(date) - last_date).days) > 5:
            ens = float(score)
            events.append({
                "date":           str(pd.Timestamp(date).date()),
                "ensemble_score": round(ens, 2),
                "risk_label":     risk_label(ens),
            })
            last_date = pd.Timestamp(date)
        if len(events) >= top_n:
            break

    # Build chart_data — merge scores with price/vol/drawdown features
    chart_data = []
    if features_path.exists():
        feat = pd.read_parquet(features_path)
        feat.index = pd.to_datetime(feat.index)
        merged = df.join(feat[["Close", "vol_30", "drawdown"]], how="left")
        for date, row in merged.iterrows():
            chart_data.append({
                "date":           str(date.date()),
                "close":          round(float(row["Close"]),         2) if pd.notna(row.get("Close")) else None,
                "ensemble_score": round(float(row["ensemble_score"]), 2),
                "vol_30":         round(float(row["vol_30"]),         4) if pd.notna(row.get("vol_30")) else None,
                "drawdown":       round(float(row["drawdown"]),       4) if pd.notna(row.get("drawdown")) else None,
                "is_anomaly":     bool(row["ensemble_score"] >= threshold),
            })
    else:
        for date, row in df.iterrows():
            chart_data.append({
                "date":           str(date.date()),
                "close":          None,
                "ensemble_score": round(float(row["ensemble_score"]), 2),
                "vol_30":         None,
                "drawdown":       None,
                "is_anomaly":     bool(row["ensemble_score"] >= threshold),
            })

    return {
        "asset":             asset,
        "events":            events,
        "total_anomaly_days": int((df["ensemble_score"] >= threshold).sum()),
        "chart_data":        chart_data,
    }


# ── 4. Model Comparison ────────────────────────────────────────────────────────
def model_comparison(asset: str) -> dict:
    """Return correlation & score stats for all 4 models."""
    scores_path = MODELS_DIR / asset / "scores_all.parquet"
    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)

    score_cols = ["zscore_score", "iforest_score", "lstm_score",
                  "prophet_score", "ensemble_score"]
    stats = {}
    for col in score_cols:
        if col not in df.columns:
            continue
        s = df[col] * (100 if col != "ensemble_score" else 1)
        stats[col] = {
            "mean":   round(float(s.mean()), 2),
            "std":    round(float(s.std()), 2),
            "max":    round(float(s.max()), 2),
            "p95":    round(float(s.quantile(0.95)), 2),
        }

    corr = df[score_cols].corr()["ensemble_score"].drop("ensemble_score")
    return {
        "asset": asset,
        "stats": stats,
        "correlation_with_ensemble": {
            col: round(float(v), 4) for col, v in corr.items()
        },
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    asset = sys.argv[1] if len(sys.argv) > 1 else "SP500"
    log.info(f"\n{'═'*60}\n  predict.py — {asset}\n{'═'*60}")

    print("\n── Current Analysis ──")
    print(json.dumps(current_analysis(asset), indent=2))

    print("\n── 5-Day Forecast ──")
    print(json.dumps(forecast_anomaly(asset, days=5), indent=2))

    print("\n── Top Historical Anomalies ──")
    hist = historical_anomalies(asset, top_n=10)
    print(json.dumps(hist, indent=2))

    print("\n── Model Comparison ──")
    print(json.dumps(model_comparison(asset), indent=2))
