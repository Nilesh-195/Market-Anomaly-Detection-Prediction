"""Utility functions for anomaly quality metrics and explainability-friendly payloads."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "backend" / "models"
DATA_DIR = ROOT_DIR / "backend" / "data" / "processed"
CRASH_LABELS_PATH = ROOT_DIR / "backend" / "data" / "crash_labels.json"


def _risk_label(score: float) -> str:
    if score < 40:
        return "Normal"
    if score < 60:
        return "Elevated"
    if score < 75:
        return "High Risk"
    return "Extreme Anomaly"


def load_crash_labels() -> dict:
    with open(CRASH_LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_scores(asset: str) -> pd.DataFrame:
    path = MODELS_DIR / asset / "scores_all.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found for {asset}: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_features(asset: str) -> pd.DataFrame:
    path = DATA_DIR / f"{asset}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features file not found for {asset}: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _resolve_model_column(df: pd.DataFrame, model: str) -> str:
    aliases = {
        "zscore": "zscore_score",
        "iforest": "iforest_score",
        "lstm": "lstm_score",
        "prophet": "prophet_score",
        "xgb": "xgb_score",
        "hmm": "hmm_score",
        "tcn": "tcn_score",
        "vae": "vae_score",
        "at": "at_score",
        "ensemble": "ensemble_score",
        "advanced": "adv_ensemble",
    }

    candidate = (model or "ensemble_score").strip().lower()
    if candidate in aliases:
        candidate = aliases[candidate]

    if candidate in df.columns:
        return candidate

    if candidate == "adv_ensemble" and "ensemble_score" in df.columns:
        return "ensemble_score"

    raise ValueError(
        f"Unknown model '{model}'. Available columns: {sorted(df.columns.tolist())}"
    )


def _normalize_score_series(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if float(s.quantile(0.99)) <= 1.5:
        s = s * 100.0
    return s.clip(0.0, 100.0)


def _asset_events(asset: str, crash_data: dict) -> list[dict]:
    events = crash_data.get("events", [])
    asset_upper = asset.upper()
    out = [e for e in events if asset_upper in e.get("assets_affected", [])]
    out.sort(key=lambda e: e.get("date", ""))
    return out


def _label_event_window(index: pd.DatetimeIndex, events: list[dict], window_days: int) -> np.ndarray:
    labels = np.zeros(len(index), dtype=int)
    if not events:
        return labels

    for event in events:
        event_date = pd.Timestamp(event["date"])
        delta = np.abs((index - event_date).days)
        labels[(delta <= window_days)] = 1
    return labels


def _confusion(labels: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _precision_recall_f1(conf: dict) -> tuple[float, float, float]:
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 5) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        left, right = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_start": round(left, 3),
                    "bin_end": round(right, 3),
                    "pred_mean": None,
                    "event_rate": None,
                    "count": 0,
                }
            )
            continue

        rows.append(
            {
                "bin_start": round(left, 3),
                "bin_end": round(right, 3),
                "pred_mean": round(float(probs[mask].mean()), 4),
                "event_rate": round(float(labels[mask].mean()), 4),
                "count": count,
            }
        )
    return rows


def _lead_time_distribution(
    events: list[dict],
    score_series: pd.Series,
    threshold: float,
    window_days: int,
) -> dict:
    categories = {
        "early_gt_7": 0,
        "early_4_7": 0,
        "early_1_3": 0,
        "same_day": 0,
        "late": 0,
        "missed": 0,
    }
    details = []

    alerted = score_series[score_series >= threshold]

    for event in events:
        event_date = pd.Timestamp(event["date"])
        in_window = alerted.loc[
            (alerted.index >= event_date - pd.Timedelta(days=window_days))
            & (alerted.index <= event_date + pd.Timedelta(days=window_days))
        ]

        if in_window.empty:
            categories["missed"] += 1
            details.append(
                {
                    "event": event.get("event", "Unknown"),
                    "event_date": str(event_date.date()),
                    "trigger_date": None,
                    "lead_days": None,
                    "status": "missed",
                }
            )
            continue

        trigger_date = in_window.index.min()
        lead_days = int((event_date - trigger_date).days)

        if lead_days > 7:
            status = "early_gt_7"
        elif lead_days >= 4:
            status = "early_4_7"
        elif lead_days >= 1:
            status = "early_1_3"
        elif lead_days == 0:
            status = "same_day"
        else:
            status = "late"

        categories[status] += 1
        details.append(
            {
                "event": event.get("event", "Unknown"),
                "event_date": str(event_date.date()),
                "trigger_date": str(trigger_date.date()),
                "lead_days": lead_days,
                "status": status,
                "trigger_score": round(float(in_window.iloc[0]), 2),
            }
        )

    return {
        "distribution": categories,
        "event_details": details,
        "n_events": len(events),
    }


def anomaly_metrics(
    asset: str,
    model: str = "ensemble_score",
    threshold: float = 60.0,
    window_days: int = 7,
) -> dict:
    df = _load_scores(asset)
    crash_data = load_crash_labels()
    events = _asset_events(asset, crash_data)

    model_col = _resolve_model_column(df, model)
    score_series = _normalize_score_series(df[model_col])
    labels = _label_event_window(df.index, events, window_days)

    probs = (score_series.values / 100.0).clip(0.0, 1.0)
    preds = (score_series.values >= threshold).astype(int)

    confusion = _confusion(labels, preds)
    precision, recall, f1 = _precision_recall_f1(confusion)

    if int(labels.sum()) > 0:
        auc_pr = float(average_precision_score(labels, probs))
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, probs)
        pr_curve = [
            {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "threshold": round(float(t) * 100.0, 2),
            }
            for p, r, t in zip(pr_precision[1:], pr_recall[1:], pr_thresholds)
        ]
    else:
        auc_pr = None
        pr_curve = []

    brier = float(np.mean((probs - labels) ** 2))
    calibration = _calibration_bins(probs, labels, n_bins=5)
    lead_time = _lead_time_distribution(events, score_series, threshold, window_days)

    return {
        "asset": asset,
        "model": model_col,
        "threshold": threshold,
        "window_days": window_days,
        "n_samples": int(len(df)),
        "event_days": int(labels.sum()),
        "events_for_asset": int(len(events)),
        "score_summary": {
            "mean": round(float(score_series.mean()), 2),
            "std": round(float(score_series.std()), 2),
            "p90": round(float(score_series.quantile(0.90)), 2),
            "p95": round(float(score_series.quantile(0.95)), 2),
            "max": round(float(score_series.max()), 2),
        },
        "classification": {
            **confusion,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        },
        "auc_pr": None if auc_pr is None else round(float(auc_pr), 4),
        "brier_score": round(float(brier), 6),
        "calibration": calibration,
        "precision_recall_curve": pr_curve,
        "lead_time": lead_time,
    }


def threshold_analysis(
    asset: str,
    model: str = "ensemble_score",
    min_threshold: float = 40.0,
    max_threshold: float = 80.0,
    step: float = 2.0,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    window_days: int = 7,
) -> dict:
    if step <= 0:
        raise ValueError("step must be > 0")
    if min_threshold >= max_threshold:
        raise ValueError("min_threshold must be < max_threshold")

    df = _load_scores(asset)
    crash_data = load_crash_labels()
    events = _asset_events(asset, crash_data)

    model_col = _resolve_model_column(df, model)
    score_series = _normalize_score_series(df[model_col])
    labels = _label_event_window(df.index, events, window_days)

    rows = []
    for threshold in np.arange(min_threshold, max_threshold + 1e-9, step):
        preds = (score_series.values >= threshold).astype(int)
        conf = _confusion(labels, preds)
        precision, recall, f1 = _precision_recall_f1(conf)
        utility = conf["tp"] - (cost_fp * conf["fp"]) - (cost_fn * conf["fn"])

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "utility": round(float(utility), 2),
                "tp": conf["tp"],
                "fp": conf["fp"],
                "tn": conf["tn"],
                "fn": conf["fn"],
            }
        )

    best = max(rows, key=lambda x: (x["utility"], x["f1"], x["precision"])) if rows else None

    return {
        "asset": asset,
        "model": model_col,
        "window_days": window_days,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "best": best,
        "thresholds": rows,
    }


def false_positive_timeline(
    asset: str,
    model: str = "ensemble_score",
    threshold: float = 60.0,
    top_n: int = 40,
    window_days: int = 7,
) -> dict:
    df = _load_scores(asset)
    crash_data = load_crash_labels()
    events = _asset_events(asset, crash_data)

    model_col = _resolve_model_column(df, model)
    scores = _normalize_score_series(df[model_col])
    labels = _label_event_window(df.index, events, window_days)

    fp_mask = (scores.values >= threshold) & (labels == 0)
    fp_df = pd.DataFrame({"score": scores}, index=df.index)
    fp_df = fp_df.loc[fp_mask]

    if top_n and top_n > 0:
        fp_df = fp_df.sort_values("score", ascending=False).head(top_n)

    fp_df = fp_df.sort_index()

    events_out = [
        {
            "date": str(idx.date()),
            "score": round(float(row["score"]), 2),
            "risk_label": _risk_label(float(row["score"])),
        }
        for idx, row in fp_df.iterrows()
    ]

    return {
        "asset": asset,
        "model": model_col,
        "threshold": threshold,
        "window_days": window_days,
        "count": len(events_out),
        "events": events_out,
    }


def bubble_risk_snapshot(asset: str) -> dict:
    feat = _load_features(asset)

    if "bubble_score" not in feat.columns:
        raise ValueError(f"bubble_score not found in features for {asset}")

    latest = feat.iloc[-1]

    bubble_series = pd.to_numeric(feat["bubble_score"], errors="coerce").fillna(0.0)
    vol_series = pd.to_numeric(feat.get("vol_30", pd.Series(index=feat.index, data=np.nan)), errors="coerce")
    drawdown_series = pd.to_numeric(feat.get("drawdown", pd.Series(index=feat.index, data=np.nan)), errors="coerce")

    bubble_raw = float(latest.get("bubble_score", 0.0))
    bubble_percentile = float((bubble_series <= bubble_raw).mean())

    vol_30 = float(latest.get("vol_30", np.nan))
    drawdown = float(latest.get("drawdown", np.nan))
    rsi_14 = float(latest.get("rsi_14", np.nan))

    vol_baseline = float(vol_series.quantile(0.5)) if vol_series.notna().any() else 20.0
    vol_component = float(np.clip((vol_baseline - vol_30) / max(vol_baseline, 1.0), 0.0, 1.0)) if np.isfinite(vol_30) else 0.5

    near_peak_component = 1.0
    if np.isfinite(drawdown):
        near_peak_component = float(np.clip(1.0 - (abs(min(drawdown, 0.0)) / 20.0), 0.0, 1.0))

    rsi_component = float(np.clip((rsi_14 - 60.0) / 40.0, 0.0, 1.0)) if np.isfinite(rsi_14) else 0.0

    risk_score = (
        0.60 * bubble_percentile
        + 0.20 * vol_component
        + 0.15 * near_peak_component
        + 0.05 * rsi_component
    ) * 100.0
    risk_score = float(np.clip(risk_score, 0.0, 100.0))

    p80 = float(bubble_series.quantile(0.80))
    p90 = float(bubble_series.quantile(0.90))
    p95 = float(bubble_series.quantile(0.95))

    return {
        "asset": asset,
        "date": str(feat.index[-1].date()),
        "bubble_score": round(bubble_raw, 4),
        "bubble_percentile": round(bubble_percentile, 4),
        "bubble_risk": round(risk_score, 2),
        "risk_label": _risk_label(risk_score),
        "is_bubble_watch": bool(risk_score >= 60.0),
        "signals": {
            "vol_30": None if not np.isfinite(vol_30) else round(vol_30, 4),
            "drawdown": None if not np.isfinite(drawdown) else round(drawdown, 4),
            "rsi_14": None if not np.isfinite(rsi_14) else round(rsi_14, 4),
        },
        "thresholds": {
            "bubble_p80": round(p80, 4),
            "bubble_p90": round(p90, 4),
            "bubble_p95": round(p95, 4),
        },
        "explanation": {
            "bubble_percentile_component": round(0.60 * bubble_percentile * 100.0, 2),
            "low_volatility_component": round(0.20 * vol_component * 100.0, 2),
            "near_peak_component": round(0.15 * near_peak_component * 100.0, 2),
            "rsi_component": round(0.05 * rsi_component * 100.0, 2),
        },
    }
