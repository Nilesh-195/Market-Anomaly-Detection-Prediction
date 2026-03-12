"""
evaluate.py
===========
Evaluates all 4 anomaly detection models + ensemble against
the 13 labelled crash events in crash_labels.json.

Metrics computed per model per asset:
  - Precision, Recall, F1  (at multiple thresholds)
  - ROC-AUC
  - Average early-warning lead (days before crash date first flagged)
  - Hit rate  (% of 13 events detected within ±5-day window)

Output:
  backend/models/evaluation_report.json   ← full metrics
  backend/models/evaluation_summary.csv   ← human-readable table

Run:
    python backend/src/evaluate.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT_DIR   = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "backend" / "models"
DATA_DIR   = ROOT_DIR / "backend" / "data"

sys.path.insert(0, str(ROOT_DIR / "backend" / "src"))

ASSETS    = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]
WINDOW    = 5          # ±5 trading days around a crash counts as a hit
THRESHOLD = 60.0       # ensemble score ≥ 60 → anomaly flag


# ── helpers ───────────────────────────────────────────────────────────────────
def load_crash_labels() -> list[dict]:
    path = DATA_DIR / "crash_labels.json"
    with open(path) as f:
        data = json.load(f)
    # Support both flat list and nested {"events": [...]} format
    if isinstance(data, list):
        return data
    return data.get("events", [])


def build_ground_truth(index: pd.Index, crash_dates: list[str],
                       window: int = WINDOW) -> np.ndarray:
    """Binary array: 1 if date is within `window` days of a crash."""
    gt = np.zeros(len(index), dtype=int)
    idx_dates = pd.to_datetime(index)
    for cd in crash_dates:
        crash_dt = pd.Timestamp(cd)
        for i, d in enumerate(idx_dates):
            if abs((d - crash_dt).days) <= window:
                gt[i] = 1
    return gt


def evaluate_scores(scores: pd.Series, gt: np.ndarray,
                    crash_dates: list[str]) -> dict:
    """Returns full metric dict for one score column vs ground truth."""
    s = scores.fillna(0).values / 100.0          # normalise to [0,1]
    pred = (scores.fillna(0).values >= THRESHOLD).astype(int)

    result = {}

    # ── Classification metrics at THRESHOLD=60 ────────────────────────────────
    result["precision"] = round(float(precision_score(gt, pred, zero_division=0)), 4)
    result["recall"]    = round(float(recall_score(gt, pred, zero_division=0)), 4)
    result["f1"]        = round(float(f1_score(gt, pred, zero_division=0)), 4)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    if gt.sum() > 0 and gt.sum() < len(gt):
        result["roc_auc"] = round(float(roc_auc_score(gt, s)), 4)
    else:
        result["roc_auc"] = None

    # ── Hit rate (crash detected within window) ────────────────────────────────
    hits, total = 0, 0
    lead_days   = []
    idx_dates   = pd.to_datetime(scores.index)
    for cd in crash_dates:
        crash_dt = pd.Timestamp(cd)
        # check if crash date exists in the index range
        if crash_dt < idx_dates[0] or crash_dt > idx_dates[-1]:
            continue
        total += 1
        # window around crash
        mask = (idx_dates >= crash_dt - pd.Timedelta(days=WINDOW)) & \
               (idx_dates <= crash_dt + pd.Timedelta(days=WINDOW))
        window_scores = scores[mask]
        if (window_scores >= THRESHOLD).any():
            hits += 1
            # earliest flag before crash
            before = scores[idx_dates <= crash_dt]
            flagged = before[before >= THRESHOLD]
            if not flagged.empty:
                first_flag = flagged.index[-1]
                lead = (crash_dt - pd.Timestamp(first_flag)).days
                if 0 <= lead <= 30:
                    lead_days.append(lead)

    result["hit_rate"]          = round(hits / total, 4) if total else 0.0
    result["crashes_detected"]  = hits
    result["crashes_in_range"]  = total
    result["avg_lead_days"]     = round(float(np.mean(lead_days)), 1) if lead_days else 0.0
    return result


# ── main evaluation loop ───────────────────────────────────────────────────────
def run_evaluation():
    crashes       = load_crash_labels()
    crash_dates   = [c["date"] for c in crashes]
    log.info(f"Loaded {len(crash_dates)} crash events")

    score_cols = ["zscore_score", "iforest_score", "lstm_score",
                  "prophet_score", "ensemble_score"]
    col_labels = {
        "zscore_score":   "Z-Score",
        "iforest_score":  "Isolation Forest",
        "lstm_score":     "LSTM Autoencoder",
        "prophet_score":  "Prophet Residual",
        "ensemble_score": "Ensemble",
    }

    report   = {}
    rows     = []        # for CSV summary

    for asset in ASSETS:
        scores_path = MODELS_DIR / asset / "scores_all.parquet"
        if not scores_path.exists():
            log.warning(f"[{asset}] scores_all.parquet not found — skipping")
            continue

        df    = pd.read_parquet(scores_path)
        df.index = pd.to_datetime(df.index)
        gt    = build_ground_truth(df.index, crash_dates)
        log.info(f"[{asset}] rows={len(df):,}  crash_windows={gt.sum():,}")

        report[asset] = {}
        for col in score_cols:
            if col not in df.columns:
                continue
            metrics = evaluate_scores(df[col], gt, crash_dates)
            report[asset][col] = metrics
            label = col_labels[col]
            log.info(
                f"  [{asset}] {label:<22}  "
                f"F1={metrics['f1']:.3f}  "
                f"AUC={metrics.get('roc_auc') or 0:.3f}  "
                f"HitRate={metrics['hit_rate']:.2f}  "
                f"LeadDays={metrics['avg_lead_days']:.1f}"
            )
            rows.append({
                "asset":    asset,
                "model":    label,
                "f1":       metrics["f1"],
                "precision":metrics["precision"],
                "recall":   metrics["recall"],
                "roc_auc":  metrics.get("roc_auc") or 0,
                "hit_rate": metrics["hit_rate"],
                "crashes_detected": metrics["crashes_detected"],
                "crashes_in_range": metrics["crashes_in_range"],
                "avg_lead_days":    metrics["avg_lead_days"],
            })

    # ── Save outputs ──────────────────────────────────────────────────────────
    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nSaved → {report_path}")

    summary_path = MODELS_DIR / "evaluation_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    log.info(f"Saved → {summary_path}")

    # ── Print best-performer table ─────────────────────────────────────────────
    df_sum = pd.DataFrame(rows)
    log.info("\n" + "═" * 70)
    log.info("  BEST MODEL PER ASSET (by Ensemble F1)")
    log.info("═" * 70)
    for asset in ASSETS:
        sub = df_sum[df_sum["asset"] == asset]
        if sub.empty:
            continue
        ens = sub[sub["model"] == "Ensemble"].iloc[0]
        log.info(
            f"  {asset:<8}  F1={ens['f1']:.3f}  "
            f"AUC={ens['roc_auc']:.3f}  "
            f"HitRate={ens['hit_rate']:.0%}  "
            f"Detected={int(ens['crashes_detected'])}/{int(ens['crashes_in_range'])}  "
            f"LeadDays={ens['avg_lead_days']:.1f}"
        )

    # ── Cross-asset ensemble summary ──────────────────────────────────────────
    ens_df = df_sum[df_sum["model"] == "Ensemble"]
    log.info("\n  OVERALL ENSEMBLE AVERAGES")
    log.info(f"  Mean F1       : {ens_df['f1'].mean():.3f}")
    log.info(f"  Mean AUC      : {ens_df['roc_auc'].mean():.3f}")
    log.info(f"  Mean Hit Rate : {ens_df['hit_rate'].mean():.1%}")
    log.info(f"  Mean Lead Days: {ens_df['avg_lead_days'].mean():.1f}")

    return report, df_sum


if __name__ == "__main__":
    run_evaluation()
