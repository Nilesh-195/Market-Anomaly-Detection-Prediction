"""
predict.py
==========
Generates predictions for a given asset:

  1. Current anomaly analysis  — latest ensemble score + per-model breakdown
    2. Advanced anomaly analysis — advanced ensemble + regime + dynamic 7-9 models (Phase 2)
    3. Multi-mode anomaly forecast (ensemble / advanced / dl / hybrid)
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


def bubble_label(score: float) -> str:
    """Simple and explainable bubble classification from bubble_score (% above 200-DMA)."""
    if score < 10:
        return "Normal"
    if score <= 25:
        return "Overextended"
    return "Extreme Overextension"


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
            bubble_val = float(last_feat.get("bubble_score", 0.0)) if pd.notna(last_feat.get("bubble_score", 0.0)) else 0.0
            result["price"]            = round(price, 2)
            result["price_change_pct"] = round((price / prev_price - 1) * 100, 3)
            result["zscore"]           = round(float(last_feat["zscore_20"]), 4)
            result["volatility"]       = round(float(last_feat["vol_30"]), 4)
            result["drawdown"]         = round(float(last_feat["drawdown"]), 4)
            result["bubble_score"]     = round(bubble_val, 4)
            result["bubble_label"]     = bubble_label(bubble_val)

    return result


# ── 1B. Advanced Current Analysis (Phase 2) ───────────────────────────────────
def advanced_current_analysis(asset: str) -> dict:
    """
    Return today's advanced model scores + regime state.
    Includes baseline + advanced scores with optional VAE/AT (dynamic 7-9 total).
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

    model_scores = {
        # Baseline (4)
        "zscore":  round(float(latest["zscore_score"]) * 100, 2),
        "iforest": round(float(latest["iforest_score"]) * 100, 2),
        "lstm":    round(float(latest["lstm_score"]) * 100, 2),
        "prophet": round(float(latest["prophet_score"]) * 100, 2),
        # Advanced (3)
        "xgb":     round(float(latest.get("xgb_score", 0)), 2),
        "hmm":     round(float(latest.get("hmm_score", 0)), 2),
        "tcn":     round(float(latest.get("tcn_score", 0)), 2),
    }

    # Optional advanced models (Phase 3+): include only when present in score parquet.
    if "vae_score" in latest.index and pd.notna(latest.get("vae_score")):
        model_scores["vae"] = round(float(latest.get("vae_score", 0)), 2)
    if "at_score" in latest.index and pd.notna(latest.get("at_score")):
        model_scores["at"] = round(float(latest.get("at_score", 0)), 2)

    result = {
        "asset":              asset,
        "date":               str(df.index[-1].date()),
        "baseline_ensemble":  round(float(latest["ensemble_score"]), 2),
        "advanced_ensemble":  round(adv_ens, 2),
        "score_delta":        round(adv_ens - adv_prev, 2),
        "risk_label":         risk_label(adv_ens),
        "current_regime":     str(latest.get("hmm_regime", "unknown")),
        "model_scores":       model_scores,
        "model_count":        len(model_scores),
        "models_available":   list(model_scores.keys()),
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
            bubble_val = float(last_feat.get("bubble_score", 0.0)) if pd.notna(last_feat.get("bubble_score", 0.0)) else 0.0
            result["price"]            = round(price, 2)
            result["price_change_pct"] = round((price / prev_price - 1) * 100, 3)
            result["zscore"]           = round(float(last_feat["zscore_20"]), 4)
            result["volatility"]       = round(float(last_feat["vol_30"]), 4)
            result["drawdown"]         = round(float(last_feat["drawdown"]), 4)
            result["bubble_score"]     = round(bubble_val, 4)
            result["bubble_label"]     = bubble_label(bubble_val)

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


# ── 2. Multi-Mode Anomaly Forecast ────────────────────────────────────────────
def _next_trading_dates(last_date: pd.Timestamp, days: int) -> list[str]:
    dates = []
    d = pd.Timestamp(last_date)
    for _ in range(days):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        dates.append(str(d.date()))
    return dates


def _normalize_score_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return s
    # Baseline model scores are often in [0,1], advanced scores are usually [0,100].
    if float(s.quantile(0.99)) <= 1.5:
        s = s * 100.0
    return s.clip(0.0, 100.0)


def _series_forecast_arima(series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from statsmodels.tsa.arima.model import ARIMA

    window = series.iloc[-252:] if len(series) > 252 else series
    window = window.dropna()
    if window.empty:
        return np.zeros(days), np.zeros(days), np.zeros(days)

    try:
        model = ARIMA(window, order=(2, 1, 2))
        fitted = model.fit()
        fc = np.asarray(fitted.forecast(steps=days), dtype=float)
        conf = fitted.get_forecast(steps=days).conf_int(alpha=0.2)
        lo = np.asarray(conf.iloc[:, 0], dtype=float)
        hi = np.asarray(conf.iloc[:, 1], dtype=float)
    except Exception as e:
        log.warning(f"ARIMA anomaly forecast failed ({e}); using robust naive fallback")
        last = float(window.iloc[-1])
        vol = float(window.diff().dropna().std()) if len(window) > 2 else 5.0
        vol = 5.0 if not np.isfinite(vol) or vol <= 0 else vol
        scale = np.sqrt(np.arange(1, days + 1, dtype=float))
        fc = np.full(days, last, dtype=float)
        band = 1.28 * vol * scale
        lo = fc - band
        hi = fc + band

    return np.clip(fc, 0, 100), np.clip(lo, 0, 100), np.clip(hi, 0, 100)


def _series_forecast_ets(series: pd.Series, days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    window = series.iloc[-252:] if len(series) > 252 else series
    window = window.dropna()
    if window.empty:
        return np.zeros(days), np.zeros(days), np.zeros(days)

    try:
        model = ExponentialSmoothing(
            window,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)
        fc = np.asarray(fitted.forecast(steps=days), dtype=float)

        resid = (window - fitted.fittedvalues).dropna()
        vol = float(resid.std()) if len(resid) > 3 else float(window.diff().dropna().std())
        vol = 5.0 if not np.isfinite(vol) or vol <= 0 else vol
        scale = np.sqrt(np.arange(1, days + 1, dtype=float))
        band = 1.28 * vol * scale
        lo = fc - band
        hi = fc + band
    except Exception as e:
        log.warning(f"ETS anomaly forecast failed ({e}); using ARIMA fallback")
        return _series_forecast_arima(series, days)

    return np.clip(fc, 0, 100), np.clip(lo, 0, 100), np.clip(hi, 0, 100)


def _blend_forecasts(
    forecast_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    forecast_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    weight_a: float = 0.6,
    weight_b: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fa, la, ua = forecast_a
    fb, lb, ub = forecast_b
    f = weight_a * fa + weight_b * fb
    l = weight_a * la + weight_b * lb
    u = weight_a * ua + weight_b * ub
    return np.clip(f, 0, 100), np.clip(l, 0, 100), np.clip(u, 0, 100)


def _build_mode_series(df: pd.DataFrame, mode: str) -> tuple[pd.Series, list[str], str]:
    mode = (mode or "hybrid").strip().lower()

    if mode == "ensemble":
        return (
            _normalize_score_series(df["ensemble_score"]),
            ["ensemble_score"],
            "Baseline ensemble (Z-Score + IForest + LSTM + Prophet)",
        )

    if mode == "advanced":
        if "adv_ensemble" in df.columns:
            models_used = ["adv_ensemble"]
            for c in ["xgb_score", "hmm_score", "tcn_score", "vae_score", "at_score"]:
                if c in df.columns:
                    models_used.append(c)
            return (
                _normalize_score_series(df["adv_ensemble"]),
                models_used,
                "Advanced ensemble (baseline + advanced models)",
            )
        return (
            _normalize_score_series(df["ensemble_score"]),
            ["ensemble_score"],
            "Advanced fallback: baseline ensemble",
        )

    if mode == "dl":
        weighted = []
        models_used = []
        dl_weights = [
            ("lstm_score", 0.35),
            ("tcn_score", 0.30),
            ("vae_score", 0.20),
            ("at_score", 0.15),
        ]
        weight_sum = 0.0
        for col, weight in dl_weights:
            if col in df.columns:
                weighted.append(_normalize_score_series(df[col]) * weight)
                models_used.append(col)
                weight_sum += weight

        if weighted and weight_sum > 0:
            combined = sum(weighted) / weight_sum
            return combined.dropna(), models_used, "DL composite (LSTM/TCN/VAE/Anomaly Transformer)"

        if "lstm_score" in df.columns:
            return (
                _normalize_score_series(df["lstm_score"]),
                ["lstm_score"],
                "DL fallback (LSTM only)",
            )

        return (
            _normalize_score_series(df["ensemble_score"]),
            ["ensemble_score"],
            "DL fallback: baseline ensemble",
        )

    if mode == "hybrid":
        adv_series, adv_models, _ = _build_mode_series(df, "advanced")
        dl_series, dl_models, _ = _build_mode_series(df, "dl")
        common_idx = adv_series.index.intersection(dl_series.index)

        if common_idx.empty:
            return adv_series, adv_models, "Hybrid fallback: advanced ensemble"

        regime_series = df["hmm_regime"] if "hmm_regime" in df.columns else pd.Series(index=common_idx, dtype=object)
        blended_vals = []
        for ts in common_idx:
            adv_v = float(adv_series.loc[ts])
            dl_v = float(dl_series.loc[ts])
            regime = str(regime_series.loc[ts]).lower() if ts in regime_series.index else ""

            if regime == "crisis":
                w_adv, w_dl = 0.45, 0.55
            elif regime == "bull":
                w_adv, w_dl = 0.60, 0.40
            else:
                w_adv, w_dl = 0.55, 0.45

            blended_vals.append(np.clip(w_adv * adv_v + w_dl * dl_v, 0, 100))

        blended = pd.Series(blended_vals, index=common_idx).astype(float)
        models_used = sorted(set(adv_models + dl_models + ["dynamic_blend"]))
        return blended, models_used, "Hybrid regime-aware blend (Advanced + DL)"

    raise ValueError(f"Unknown forecast mode '{mode}'")


def forecast_anomaly(asset: str, days: int = 5, mode: str = "hybrid", method: str = "hybrid") -> dict:
    """
    Forecast anomaly risk score for next `days` trading days.

    mode:
      - ensemble: baseline ensemble score
      - advanced: advanced ensemble score
      - dl: deep-learning composite score
      - hybrid: regime-aware blend of advanced and DL scores

    method:
      - arima: ARIMA projection
      - ets: exponential smoothing projection
      - hybrid: blend of ARIMA and ETS projections
    """
    mode = (mode or "hybrid").strip().lower()
    method = (method or "hybrid").strip().lower()

    scores_path = MODELS_DIR / asset / "scores_all.parquet"
    df = pd.read_parquet(scores_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    series, models_used, source_label = _build_mode_series(df, mode)
    series = series.dropna()

    if series.empty:
        raise ValueError(f"No usable anomaly score series for asset={asset}, mode={mode}")

    if method == "arima":
        f, l, u = _series_forecast_arima(series, days)
    elif method == "ets":
        f, l, u = _series_forecast_ets(series, days)
    elif method == "hybrid":
        arima_pack = _series_forecast_arima(series, days)
        ets_pack = _series_forecast_ets(series, days)
        f, l, u = _blend_forecasts(arima_pack, ets_pack, weight_a=0.6, weight_b=0.4)
    else:
        raise ValueError(f"Unknown forecast method '{method}'")

    dates = _next_trading_dates(series.index[-1], days)

    forecast_points = []
    for i in range(days):
        score = float(np.clip(f[i], 0, 100))
        lo = float(np.clip(l[i], 0, 100))
        hi = float(np.clip(u[i], 0, 100))
        forecast_points.append(
            {
                "date": dates[i],
                "score": round(score, 2),
                "lower": round(lo, 2),
                "upper": round(hi, 2),
                "risk_label": risk_label(score),
            }
        )

    return {
        "asset": asset,
        "generated": str(datetime.now().date()),
        "horizon": days,
        "mode": mode,
        "method": method,
        "source_label": source_label,
        "models_used": models_used,
        "available_modes": ["ensemble", "advanced", "dl", "hybrid"],
        "available_methods": ["arima", "ets", "hybrid"],
        "forecast": forecast_points,
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

    # Build chart_data — merge scores with price/vol/drawdown/bubble features
    chart_data = []
    if features_path.exists():
        feat = pd.read_parquet(features_path)
        feat.index = pd.to_datetime(feat.index)
        merged = df.join(feat[["Close", "vol_30", "drawdown", "bubble_score"]], how="left")
        for date, row in merged.iterrows():
            chart_data.append({
                "date":           str(date.date()),
                "close":          round(float(row["Close"]),         2) if pd.notna(row.get("Close")) else None,
                "ensemble_score": round(float(row["ensemble_score"]), 2),
                "vol_30":         round(float(row["vol_30"]),         4) if pd.notna(row.get("vol_30")) else None,
                "drawdown":       round(float(row["drawdown"]),       4) if pd.notna(row.get("drawdown")) else None,
                "bubble_score":   round(float(row["bubble_score"]),   4) if pd.notna(row.get("bubble_score")) else None,
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
                "bubble_score":   None,
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
