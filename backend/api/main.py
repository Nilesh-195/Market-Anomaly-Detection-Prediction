"""
main.py — FastAPI backend for Market Anomaly Detection & Prediction
====================================================================
Endpoints:
  GET  /                            → health check
  GET  /assets                      → list of supported assets
  GET  /current-analysis/{asset}    → latest anomaly score + breakdown
  GET  /forecast/{asset}            → 5-day ARIMA anomaly score forecast
  GET  /historical-anomalies/{asset}→ top historical anomaly events
  GET  /model-comparison/{asset}    → per-model stats + correlations
  GET  /evaluation                  → full evaluation report (all assets)

Run:
    uvicorn backend.api.main:app --reload --port 8000
"""

import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "backend" / "src"))

from predict import (
    current_analysis,
    forecast_anomaly,
    historical_anomalies,
    model_comparison,
    ASSETS,
)

log = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Market Anomaly Detection API",
    description="Real-time market anomaly detection using Z-Score, Isolation Forest, LSTM Autoencoder, and Prophet.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_asset(asset: str):
    upper = asset.upper()
    if upper not in ASSETS:
        raise HTTPException(
            status_code=404,
            detail=f"Asset '{asset}' not found. Valid assets: {ASSETS}",
        )
    return upper


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "Market Anomaly Detection API is running."}


# ── Assets ─────────────────────────────────────────────────────────────────────
@app.get("/assets")
def list_assets():
    return {
        "assets": ASSETS,
        "descriptions": {
            "SP500":  "S&P 500 Index (^GSPC)",
            "VIX":    "CBOE Volatility Index (^VIX)",
            "BTC":    "Bitcoin / USD (BTC-USD)",
            "GOLD":   "Gold ETF (GLD)",
            "NASDAQ": "NASDAQ 100 ETF (QQQ)",
            "TESLA":  "Tesla Inc. (TSLA)",
        },
    }


# ── Current Analysis ───────────────────────────────────────────────────────────
@app.get("/current-analysis/{asset}")
def get_current_analysis(asset: str):
    """Latest anomaly score, risk label, and per-model breakdown."""
    a = _check_asset(asset)
    try:
        return current_analysis(a)
    except Exception as e:
        log.error(f"current_analysis error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Forecast ───────────────────────────────────────────────────────────────────
@app.get("/forecast/{asset}")
def get_forecast(asset: str, days: int = 5):
    """ARIMA 5-day anomaly score forecast with confidence intervals."""
    a = _check_asset(asset)
    if not 1 <= days <= 30:
        raise HTTPException(status_code=400, detail="days must be between 1 and 30")
    try:
        return forecast_anomaly(a, days=days)
    except Exception as e:
        log.error(f"forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Historical Anomalies ───────────────────────────────────────────────────────
@app.get("/historical-anomalies/{asset}")
def get_historical_anomalies(asset: str, top_n: int = 20, threshold: float = 60.0):
    """Top historical anomaly events clustered into 5-day windows."""
    a = _check_asset(asset)
    try:
        return historical_anomalies(a, top_n=top_n, threshold=threshold)
    except Exception as e:
        log.error(f"historical_anomalies error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Model Comparison ───────────────────────────────────────────────────────────
@app.get("/model-comparison/{asset}")
def get_model_comparison(asset: str):
    """Per-model stats and correlation with ensemble score."""
    a = _check_asset(asset)
    try:
        return model_comparison(a)
    except Exception as e:
        log.error(f"model_comparison error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Evaluation Report ──────────────────────────────────────────────────────────
@app.get("/evaluation")
def get_evaluation():
    """Full evaluation report (Precision/Recall/F1/AUC per model per asset)."""
    report_path = ROOT_DIR / "backend" / "models" / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404,
                            detail="Evaluation report not found. Run evaluate.py first.")
    with open(report_path) as f:
        return json.load(f)


# ── All Assets Summary ─────────────────────────────────────────────────────────
@app.get("/summary")
def get_summary():
    """Current analysis for all assets — dashboard overview."""
    results = []
    for asset in ASSETS:
        try:
            results.append(current_analysis(asset))
        except Exception as e:
            results.append({"asset": asset, "error": str(e)})
    return {"assets": results}
