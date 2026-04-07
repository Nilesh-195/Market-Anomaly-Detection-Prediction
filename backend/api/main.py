"""
main.py — FastAPI backend for Time Series Forecasting & Anomaly Detection
===========================================================================
PRIMARY: Stock Price Forecasting using classical TSFA methods
BONUS: Market Anomaly Detection (Phase 2: 9-model advanced ensemble)

FORECASTING ENDPOINTS:
  GET  /forecast/price/{asset}           → Price forecast (best method)
  GET  /forecast/stationarity/{asset}    → ADF/KPSS stationarity tests
  GET  /forecast/acf-pacf/{asset}        → ACF/PACF plots + ARIMA order suggestion
  GET  /forecast/naive/{asset}           → All 4 naive method forecasts
  GET  /forecast/exponential/{asset}     → Exponential smoothing methods
  GET  /forecast/arima/{asset}           → ARIMA/SARIMA forecasts
  GET  /forecast/compare/{asset}         → Compare all methods side-by-side
  GET  /forecast/var                     → Multi-asset VAR forecast
  GET  /forecast/evaluate/{asset}        → CV evaluation results

ANOMALY DETECTION ENDPOINTS:
  Baseline (4 models):
    GET  /anomaly/current/{asset}        → Latest anomaly score (baseline)
    GET  /anomaly/forecast/{asset}       → Anomaly score forecast
    GET  /anomaly/historical/{asset}     → Historical anomaly events
    GET  /anomaly/comparison/{asset}     → Per-model anomaly stats
    Advanced (Phase 2 - 9 models):
        GET  /anomaly/advanced/{asset}       → 9-model advanced view + ensemble
    GET  /anomaly/regime/{asset}         → HMM market regime timeline
    GET  /anomaly/compare-tiers/{asset}  → Baseline vs Advanced comparison

GENERAL:
  GET  /                                 → Health check + API info
  GET  /assets                           → List supported assets
  GET  /summary                          → Dashboard overview (all assets)

Run:
    uvicorn backend.api.main:app --reload --port 8000
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "backend" / "src"))

# Import forecasting modules (NEW - PRIMARY)
from stationarity import analyze_stationarity, adf_test, comprehensive_stationarity_test
from acf_pacf_analysis import analyze_acf_pacf, suggest_arima_order
from naive_methods import run_all_naive_methods
from exponential_smoothing import run_all_exp_smoothing
from arima_models import arima_forecast, sarima_forecast
from var_model import analyze_var, load_multi_asset_data
from forecast_evaluation import evaluate_all_methods
from features import load_all_features
from dl_models import lstm_seq2seq_forecast, transformer_forecast
from gb_models import xgboost_forecast  # Phase 3 DL
from anomaly_quality import (
    load_crash_labels,
    anomaly_metrics,
    threshold_analysis as anomaly_threshold_analysis,
    false_positive_timeline,
    bubble_risk_snapshot,
)

# Import anomaly detection (LEGACY - SECONDARY)
from predict import (
    current_analysis,
    advanced_current_analysis,  # Phase 2
    regime_timeline,            # Phase 2
    forecast_anomaly,
    historical_anomalies,
    model_comparison,
    ASSETS,
)

log = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Time Series Forecasting & Anomaly Detection API",
    description=(
        "Production-grade stock price forecasting system using classical TSFA methods:\n\n"
        "**PRIMARY FEATURES:**\n"
        "- Price forecasting with Naive, Exponential Smoothing, ARIMA/SARIMA, VAR\n"
        "- Stationarity analysis (ADF/KPSS tests)\n"
        "- ACF/PACF analysis for ARIMA order selection\n"
        "- Cross-validation and method comparison\n"
        "- Prediction intervals with 95% confidence bands\n\n"
        "**BONUS FEATURES (Phase 2 Upgraded):**\n"
        "- Market anomaly detection with 9-model advanced view:\n"
        "  * Baseline: Z-Score, Isolation Forest, LSTM, Prophet\n"
        "  * Advanced: XGBoost (supervised), HMM (regime), TCN (temporal), VAE, Anomaly Transformer\n"
        "- Market regime detection (bull/bear/crisis)\n"
        "- Historical crash event analysis (24 labeled events)\n"
        "- Advanced ensemble with macro features"
    ),
    version="2.1.0",
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


def _format_dates(index, train_end_date=None, horizon=None):
    """Safely format dates from forecast index."""
    dates = []
    for i, d in enumerate(index):
        if hasattr(d, 'date'):
            dates.append(str(d.date()))
        elif hasattr(d, 'strftime'):
            dates.append(d.strftime('%Y-%m-%d'))
        elif train_end_date is not None:
            # Generate future dates from train end date
            next_date = train_end_date + pd.Timedelta(days=i + 1)
            dates.append(str(next_date.date()))
        else:
            dates.append(str(d))
    return dates


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def health():
    """API health check and information."""
    return {
        "status": "ok",
        "message": "Time Series Forecasting & Anomaly Detection API",
        "version": "2.1.0",
        "primary_features": [
            "Stock price forecasting (Naive, Exp. Smoothing, ARIMA, VAR)",
            "Stationarity testing (ADF/KPSS)",
            "ACF/PACF analysis",
            "Method comparison & evaluation",
            "Prediction intervals",
        ],
        "bonus_features": [
            "Anomaly detection",
            "Historical crash events",
        ],
        "supported_assets": ASSETS,
        "endpoints": {
            "forecasting": "/docs#/Forecasting",
            "anomaly_detection": "/docs#/Anomaly%20Detection",
            "general": "/docs#/General",
            "major_events": "/events/crashes",
        },
    }


@app.get("/assets", tags=["General"])
def list_assets():
    """List all supported assets with descriptions."""
    return {
        "assets": ASSETS,
        "count": len(ASSETS),
        "descriptions": {
            "SP500":  "S&P 500 Index (^GSPC)",
            "VIX":    "CBOE Volatility Index (^VIX)",
            "BTC":    "Bitcoin / USD (BTC-USD)",
            "GOLD":   "Gold ETF (GLD)",
            "NASDAQ": "NASDAQ 100 ETF (QQQ)",
            "TESLA":  "Tesla Inc. (TSLA)",
        },
        "data_available": {
            "price_history": "2010-01-01 to present",
            "features": "15 engineered features per asset",
            "forecasting_models": ["Naive", "SES", "Holt", "ARIMA", "SARIMA"],
        },
    }


@app.get("/summary", tags=["General"])
def get_summary():
    """
    Dashboard overview - Current status for all assets.

    Returns latest price, 1-day forecast, anomaly score for each asset.
    """
    results = []
    features_data = load_all_features()

    for asset in ASSETS:
        try:
            if asset not in features_data:
                results.append({"asset": asset, "error": "Data not found"})
                continue

            df = features_data[asset]
            latest = df.iloc[-1]
            current_price = float(latest["Close"])

            # Quick 1-day naive forecast
            forecast_1d = current_price  # Naive forecast

            # Get anomaly score (if available)
            try:
                anomaly_data = current_analysis(asset)
                anomaly_score = anomaly_data.get("ensemble_score", 0)
                risk_label = anomaly_data.get("risk_label", "Unknown")
            except:
                anomaly_score = 0
                risk_label = "N/A"

            results.append({
                "asset": asset,
                "current_price": round(current_price, 2),
                "forecast_1d": round(forecast_1d, 2),
                "change_1d_pct": 0,  # Naive = no change
                "anomaly_score": round(anomaly_score, 2),
                "risk_label": risk_label,
                "last_updated": str(latest.name.date()) if hasattr(latest.name, 'date') else "N/A",
            })

        except Exception as e:
            log.error(f"Summary error for {asset}: {e}")
            results.append({"asset": asset, "error": str(e)})

    return {
        "assets": results,
        "count": len(results),
        "timestamp": str(pd.Timestamp.now()),
    }


@app.get("/events/crashes", tags=["General"])
def get_crash_events(asset: str = None, from_date: str = None, to_date: str = None):
    """
    Return labeled major crash events from crash_labels.json.

    Used by frontend major-event markers and event timeline widgets.

    Optional filters:
    - asset: keep only events that affected this asset
    - from_date / to_date: ISO date bounds (YYYY-MM-DD)
    """
    labels_path = ROOT_DIR / "backend" / "data" / "crash_labels.json"
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="crash_labels.json not found")

    try:
        with open(labels_path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        log.error(f"events/crashes read error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    events = payload.get("events", [])

    if asset:
        asset_upper = asset.upper()
        if asset_upper not in ASSETS:
            raise HTTPException(status_code=404, detail=f"Asset '{asset}' not found. Valid assets: {ASSETS}")
        events = [e for e in events if asset_upper in e.get("assets_affected", [])]

    start_ts = None
    end_ts = None

    if from_date:
        try:
            start_ts = pd.Timestamp(from_date)
        except Exception:
            raise HTTPException(status_code=400, detail="from_date must be ISO format YYYY-MM-DD")

    if to_date:
        try:
            end_ts = pd.Timestamp(to_date)
        except Exception:
            raise HTTPException(status_code=400, detail="to_date must be ISO format YYYY-MM-DD")

    if start_ts is not None or end_ts is not None:
        filtered = []
        for event in events:
            try:
                event_ts = pd.Timestamp(event.get("date"))
            except Exception:
                continue
            if start_ts is not None and event_ts < start_ts:
                continue
            if end_ts is not None and event_ts > end_ts:
                continue
            filtered.append(event)
        events = filtered

    events = sorted(events, key=lambda x: x.get("date", ""))
    return {
        "total_events": len(events),
        "events": events,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORECASTING ENDPOINTS (PRIMARY FEATURE)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/forecast/stationarity/{asset}", tags=["Forecasting"])
def forecast_stationarity(asset: str):
    """
    Stationarity analysis using ADF and KPSS tests.

    Returns:
    - ADF test results (statistic, p-value, critical values)
    - KPSS test results
    - Recommended differencing order (d)
    - Interpretation
    """
    a = _check_asset(asset)

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]

        # Comprehensive stationarity test
        result = comprehensive_stationarity_test(series, a)

        return {
            "asset": a,
            "adf_test": {
                "statistic": float(result["adf"]["adf_statistic"]),
                "p_value": float(result["adf"]["p_value"]),
                "n_lags": int(result["adf"]["n_lags"]),
                "n_obs": int(result["adf"]["n_obs"]),
                "critical_values": {k: float(v) for k, v in result["adf"]["critical_values"].items()},
                "is_stationary": bool(result["adf"]["is_stationary"]),
            },
            "kpss_test": {
                "statistic": float(result["kpss"]["kpss_statistic"]),
                "p_value": float(result["kpss"]["p_value"]),
                "critical_values": {k: float(v) for k, v in result["kpss"]["critical_values"].items()},
                "is_stationary": bool(result["kpss"]["is_stationary"]),
            },
            "interpretation": result["interpretation"],
            "recommendation": result["recommendation"],
            "needs_differencing": bool(result["needs_differencing"]),
        }

    except Exception as e:
        log.error(f"Stationarity error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/acf-pacf/{asset}", tags=["Forecasting"])
def forecast_acf_pacf(asset: str, max_lags: int = 40):
    """
    ACF/PACF analysis and ARIMA order suggestion.

    Parameters:
    - max_lags: Maximum number of lags to compute (default: 40)

    Returns:
    - ACF values with significance
    - PACF values with significance
    - Suggested ARIMA(p,d,q) order
    - Alternative orders to try
    """
    a = _check_asset(asset)

    if not 10 <= max_lags <= 100:
        raise HTTPException(status_code=400, detail="max_lags must be between 10 and 100")

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]

        # First difference the series
        differenced = series.diff().dropna()

        # Get ARIMA order suggestion
        order_result = suggest_arima_order(differenced, d=1, max_p=5, max_q=5, name=a)

        return {
            "asset": a,
            "suggested_order": {
                "p": int(order_result["suggested_p"]),
                "d": int(order_result["suggested_d"]),
                "q": int(order_result["suggested_q"]),
                "full": f"ARIMA({order_result['suggested_p']},{order_result['suggested_d']},{order_result['suggested_q']})",
            },
            "acf_significant_lags": [int(x) for x in order_result["acf_analysis"]["significant_lags"][:10]],
            "pacf_significant_lags": [int(x) for x in order_result["pacf_analysis"]["significant_lags"][:10]],
            "alternative_orders": order_result["alternatives"],
            "interpretation": order_result["interpretation"],
        }

    except Exception as e:
        log.error(f"ACF/PACF error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/naive/{asset}", tags=["Forecasting"])
def forecast_naive(asset: str, horizon: int = 30):
    """
    Naive method forecasts (all 4 baseline methods).

    Methods:
    1. Mean: forecast = mean(training data)
    2. Naive: forecast = last observed value
    3. Seasonal Naive: forecast = value from same season last period
    4. Drift: linear trend from first to last observation

    Parameters:
    - horizon: Forecast horizon in days (1-90, default: 30)

    Returns forecasts with prediction intervals for all methods.
    """
    a = _check_asset(asset)

    if not 1 <= horizon <= 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90")

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]
        train = series
        test = pd.Series([], dtype=float)

        # Run all naive methods
        results = run_all_naive_methods(train, test, seasonal_period=21, name=a, horizon=horizon)

        # Format output
        methods_output = {}
        for method_name, method_result in results["methods"].items():
            methods_output[method_name] = {
                "forecast": method_result["forecast"].tolist(),
                "lower_95": method_result["lower"].tolist(),
                "upper_95": method_result["upper"].tolist(),
                "dates": _format_dates(method_result["forecast"].index, train.index[-1]),
            }

            if "metrics" in method_result:
                methods_output[method_name]["metrics"] = {
                    "rmse": float(method_result["metrics"]["rmse"]),
                    "mae": float(method_result["metrics"]["mae"]),
                    "mape": float(method_result["metrics"]["mape"]),
                }

        return {
            "asset": a,
            "horizon": horizon,
            "methods": methods_output,
            "best_method": results["best_method"],
            "comparison": results["comparison"].to_dict(orient="records"),
        }

    except Exception as e:
        log.error(f"Naive forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/exponential/{asset}", tags=["Forecasting"])
def forecast_exponential(asset: str, horizon: int = 30, method: str = "auto"):
    """
    Exponential smoothing forecasts.

    Methods:
    - auto: Select best method by AIC
    - ses: Simple Exponential Smoothing
    - holt: Holt's Linear Trend
    - damped: Damped Trend
    - holtwinters: Holt-Winters Seasonal

    Parameters:
    - horizon: Forecast horizon in days (1-90, default: 30)
    - method: Specific method or 'auto' (default: auto)

    Returns forecast with prediction intervals and model info.
    """
    a = _check_asset(asset)

    if not 1 <= horizon <= 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90")

    valid_methods = ["auto", "ses", "holt", "damped", "holtwinters"]
    if method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"method must be one of {valid_methods}")

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]
        train = series
        test = pd.Series([], dtype=float)

        # Run exp smoothing
        results = run_all_exp_smoothing(train, test, seasonal_period=21, name=a, horizon=horizon)

        if method == "auto":
            method_name = results["best_method"]
        else:
            method_map = {
                "ses": "SES",
                "holt": "Holt",
                "damped": "Damped",
                "holtwinters": "Holt-Winters",
            }
            method_name = method_map.get(method, "Holt")
            if method_name not in results["methods"]:
                method_name = results["best_method"]

        method_result = results["methods"][method_name]

        output = {
            "asset": a,
            "horizon": horizon,
            "method": method_name,
            "forecast": method_result["forecast"].tolist(),
            "lower_95": method_result["lower"].tolist(),
            "upper_95": method_result["upper"].tolist(),
            "dates": _format_dates(method_result["forecast"].index, train.index[-1]),
            "model_info": {
                "aic": float(method_result.get("aic", 0)),
                "bic": float(method_result.get("bic", 0)),
                "params": method_result.get("params", {}),
            },
        }

        if "metrics" in method_result:
            output["metrics"] = {
                "rmse": float(method_result["metrics"]["rmse"]),
                "mae": float(method_result["metrics"]["mae"]),
                "mape": float(method_result["metrics"]["mape"]),
            }

        # Include comparison if auto
        if method == "auto":
            output["all_methods_comparison"] = results["comparison"].to_dict(orient="records")

        return output

    except Exception as e:
        log.error(f"Exp smoothing error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/arima/{asset}", tags=["Forecasting"])
def forecast_arima_endpoint(
    asset: str,
    horizon: int = 30,
    p: int = None,
    d: int = None,
    q: int = None,
    seasonal: bool = False,
):
    """
    ARIMA/SARIMA price forecast.

    Parameters:
    - horizon: Forecast horizon in days (1-90, default: 30)
    - p, d, q: ARIMA orders (auto-selected if not provided)
    - seasonal: Use SARIMA with seasonal component (default: False)

    Returns price forecast with prediction intervals and diagnostics.
    """
    a = _check_asset(asset)

    if not 1 <= horizon <= 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90")

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]
        train = series

        # Auto-select order if not provided
        if p is None or d is None or q is None:
            # Simple auto-selection
            p = p or 1
            d = d or 1
            q = q or 1

        order = (p, d, q)

        # Forecast
        if seasonal:
            result = sarima_forecast(
                train,
                horizon,
                order=order,
                seasonal_order=(1, 0, 1, 21),
                name=a
            )
        else:
            result = arima_forecast(train, horizon, order=order, name=a)

        if not result["success"]:
            raise HTTPException(status_code=500, detail="ARIMA model fitting failed")

        return {
            "asset": a,
            "horizon": horizon,
            "order": order,
            "seasonal": seasonal,
            "forecast": result["forecast"].tolist(),
            "lower_95": result["lower"].tolist(),
            "upper_95": result["upper"].tolist(),
            "dates": _format_dates(result["forecast"].index, train.index[-1], horizon),
            "model_info": {
                "aic": float(result["aic"]),
                "bic": float(result["bic"]),
                "method": result["method"],
            },
        }

    except Exception as e:
        log.error(f"ARIMA error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/compare/{asset}", tags=["Forecasting"])
def forecast_compare(asset: str, horizon: int = 30):
    """
    Compare all forecasting methods side-by-side.

    Runs: Naive, Mean, Drift, SES, Holt, ARIMA and returns
    comparison table ranked by RMSE.

    Parameters:
    - horizon: Forecast horizon in days (1-90, default: 30)

    Returns ranked comparison with metrics for all methods.
    """
    a = _check_asset(asset)

    if not 1 <= horizon <= 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90")

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]

        # Run comprehensive evaluation
        eval_results = evaluate_all_methods(series, a, test_size=horizon, cv_folds=3, save_plots=False)

        # Extract comparison table
        comparison = eval_results.get("comparison", pd.DataFrame())

        if comparison.empty:
            raise HTTPException(status_code=500, detail="Evaluation failed")

        return {
            "asset": a,
            "horizon": horizon,
            "comparison": comparison.to_dict(orient="records"),
            "best_method": eval_results.get("best_method", "Unknown"),
            "best_rmse": float(eval_results.get("best_rmse", 0)),
        }

    except Exception as e:
        log.error(f"Compare error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/price/{asset}", tags=["Forecasting"])
def forecast_price(asset: str, horizon: int = 30, method: str = "auto"):
    """
    **PRIMARY ENDPOINT**: Get best price forecast for an asset.

    This is the main forecasting endpoint that automatically selects
    the best method based on CV evaluation.

    Parameters:
    - horizon: Forecast horizon in days (1-90, default: 30)
    - method: Specific method or 'auto' to select best (default: auto)

    Methods available: naive, mean, drift, ses, holt, arima, lstm, transformer, xgboost

    Returns:
    - Price forecast with 95% CI
    - Method used
    - Forecast accuracy metrics
    """
    a = _check_asset(asset)

    if not 1 <= horizon <= 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90")

    try:
        features_data = load_all_features()
        if a not in features_data:
            raise HTTPException(status_code=404, detail=f"Data for {a} not found")

        series = features_data[a]["Close"]
        train = series
        test = pd.Series([], dtype=float)

        # If auto, determine best method via quick comparison
        if method == "auto":
            # Quick comparison (no full CV to save time)
            from naive_methods import naive_forecast
            from exponential_smoothing import holt_forecast
            from arima_models import arima_forecast as arima_fcst

            # Try top 3 methods
            candidates = {}

            try:
                nf = naive_forecast(train, min(10, horizon))
                candidates["naive"] = nf
            except:
                pass

            try:
                hf = holt_forecast(train, min(10, horizon))
                candidates["holt"] = hf
            except:
                pass

            try:
                af = arima_fcst(train, min(10, horizon), order=(1,1,1), name=a)
                if af["success"]:
                    candidates["arima"] = af
            except:
                pass

            # Select best by AIC (if available) or default to ARIMA
            best_method = "arima"
            best_aic = float('inf')
            for m, res in candidates.items():
                aic = res.get("aic", float('inf'))
                if aic < best_aic:
                    best_aic = aic
                    best_method = m

            method = best_method

        # Generate forecast with selected method
        if method in ["naive", "mean", "drift"]:
            result_dict = forecast_naive(a, horizon)
            naive_result = result_dict["methods"][method]
            result = {
                "forecast": pd.Series(naive_result["forecast"]),
                "lower": pd.Series(naive_result["lower_95"]),
                "upper": pd.Series(naive_result["upper_95"]),
            }
            model_info = {"method": method, "category": "naive"}
        elif method in ["ses", "holt"]:
            result_dict = forecast_exponential(a, horizon, method=method)
            result = {
                "forecast": pd.Series(result_dict["forecast"]),
                "lower": pd.Series(result_dict["lower_95"]),
                "upper": pd.Series(result_dict["upper_95"]),
            }
            model_info = result_dict.get("model_info", {})
            model_info["category"] = "exponential_smoothing"
        elif method == "lstm":
            result_dict = lstm_seq2seq_forecast(a, horizon=horizon)
            result = {
                "forecast": pd.Series(result_dict["forecast"]),
                "lower": pd.Series(result_dict["lower_95"]),
                "upper": pd.Series(result_dict["upper_95"]),
            }
            model_info = result_dict.get("model_info", {})
            model_info["category"] = "deep_learning"
        elif method == "transformer":
            result_dict = transformer_forecast(a, horizon=horizon)
            result = {
                "forecast": pd.Series(result_dict["forecast"]),
                "lower": pd.Series(result_dict["lower_95"]),
                "upper": pd.Series(result_dict["upper_95"]),
            }
            model_info = result_dict.get("model_info", {})
            model_info["category"] = "deep_learning"
        elif method == "xgboost":
            result_dict = xgboost_forecast(a, horizon=horizon)
            result = {
                "forecast": pd.Series(result_dict["forecast"]),
                "lower": pd.Series(result_dict["lower_95"]),
                "upper": pd.Series(result_dict["upper_95"]),
            }
            model_info = result_dict.get("model_info", {})
            model_info["category"] = "gradient_boosting"
        else:  # arima (default)
            result_dict = forecast_arima_endpoint(a, horizon)
            result = {
                "forecast": pd.Series(result_dict["forecast"]),
                "lower": pd.Series(result_dict["lower_95"]),
                "upper": pd.Series(result_dict["upper_95"]),
            }
            model_info = result_dict.get("model_info", {})
            model_info["category"] = "arima"

        # Pass through visualization payload directly into response roots
        attention_weights = result_dict.get("attention_weights")
        feature_importance = result_dict.get("feature_importance")
        # For Phase 3 Transformer, feature_weights acts as feature_importance
        if method == "transformer" and "feature_weights" in result_dict:
            feature_importance = result_dict["feature_weights"]

        response_payload = {
            "asset": a,
            "current_price": float(train.iloc[-1]),
            "horizon": horizon,
            "method": method,
            "model_info": model_info,
            "attention_weights": attention_weights,
            "feature_importance": feature_importance,
            "forecast": {
                "values": result["forecast"].tolist() if hasattr(result["forecast"], "tolist") else result["forecast"],
                "lower_95": result["lower"].tolist() if hasattr(result["lower"], "tolist") else result["lower"],
                "upper_95": result["upper"].tolist() if hasattr(result["upper"], "tolist") else result["upper"],
                "dates": result_dict.get("dates", []),
            },
            "summary": {
                "forecast_30d": float(result["forecast"].iloc[-1] if hasattr(result["forecast"], "iloc") else result["forecast"][-1]),
                "expected_return_pct": round(((float(result["forecast"].iloc[-1] if hasattr(result["forecast"], "iloc") else result["forecast"][-1]) / float(train.iloc[-1])) - 1) * 100, 2),
            },
        }

        return response_payload

    except Exception as e:
        log.error(f"Price forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/var", tags=["Forecasting"])
def forecast_var(
    assets: str = "SP500,NASDAQ,VIX",
    horizon: int = 30,
):
    """
    Multi-asset VAR forecast with Granger causality.

    Parameters:
    - assets: Comma-separated list of assets (e.g., "SP500,NASDAQ,VIX")
    - horizon: Forecast horizon in days (1-90, default: 30)

    Returns:
    - Joint forecast for all assets
    - Granger causality results
    - Lag order selected
    """
    asset_list = [a.strip().upper() for a in assets.split(",")]

    # Validate all assets
    for a in asset_list:
        if a not in ASSETS:
            raise HTTPException(status_code=404, detail=f"Asset '{a}' not supported")

    if len(asset_list) < 2:
        raise HTTPException(status_code=400, detail="VAR requires at least 2 assets")

    if not 1 <= horizon <= 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90")

    try:
        # Run VAR analysis
        var_results = analyze_var(asset_list, test_size=horizon, max_lags=10, save_plots=False)

        if not var_results.get("success"):
            raise HTTPException(status_code=500, detail="VAR analysis failed")

        # Extract forecast
        forecast_result = var_results["forecast_result"]
        granger = var_results["granger_causality"]

        # Format output
        forecasts_by_asset = {}
        for asset in asset_list:
            if asset in forecast_result["forecast"].columns:
                forecasts_by_asset[asset] = {
                    "forecast": forecast_result["forecast"][asset].tolist(),
                    "lower_95": forecast_result["lower"][asset].tolist(),
                    "upper_95": forecast_result["upper"][asset].tolist(),
                    "dates": _format_dates(forecast_result["forecast"].index),
                }

        # Granger causality (significant only)
        significant_granger = granger[granger["Granger_Causes"]]

        return {
            "assets": asset_list,
            "horizon": horizon,
            "lag_order": int(var_results["var_result"]["lag_order"]),
            "forecasts": forecasts_by_asset,
            "granger_causality": significant_granger.to_dict(orient="records"),
            "model_info": {
                "aic": float(var_results["var_result"]["aic"]),
                "n_observations": int(var_results["var_result"]["n_obs"]),
            },
        }

    except Exception as e:
        log.error(f"VAR error for {assets}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION ENDPOINTS (LEGACY / BONUS FEATURE)
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/anomaly/current/{asset}", tags=["Anomaly Detection"])
def get_current_anomaly(asset: str):
    """
    Latest anomaly score and risk assessment.

    Returns ensemble anomaly score (0-100) with risk label,
    per-model breakdown (Z-Score, Isolation Forest, LSTM, Prophet),
    and market-context fields including bubble_score and bubble_label.
    """
    a = _check_asset(asset)
    try:
        return current_analysis(a)
    except Exception as e:
        log.error(f"current_analysis error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/forecast/{asset}", tags=["Anomaly Detection"])
def get_anomaly_forecast(
    asset: str,
    days: int = 5,
    mode: str = "hybrid",
    method: str = "hybrid",
):
    """
    Multi-model forecast of anomaly scores.

    Forecasts future anomaly risk scores (not prices) using baseline,
    advanced, deep-learning composite, or hybrid signal modes.

    Parameters:
    - days: Forecast horizon (1-30, default: 5)
    - mode: ensemble | advanced | dl | hybrid (default: hybrid)
    - method: arima | ets | hybrid (default: hybrid)
    """
    a = _check_asset(asset)
    if not 1 <= days <= 30:
        raise HTTPException(status_code=400, detail="days must be between 1 and 30")

    mode = (mode or "hybrid").strip().lower()
    method = (method or "hybrid").strip().lower()
    valid_modes = ["ensemble", "advanced", "dl", "hybrid"]
    valid_methods = ["arima", "ets", "hybrid"]

    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"mode must be one of {valid_modes}")
    if method not in valid_methods:
        raise HTTPException(status_code=400, detail=f"method must be one of {valid_methods}")

    try:
        return forecast_anomaly(a, days=days, mode=mode, method=method)
    except Exception as e:
        log.error(f"forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/historical/{asset}", tags=["Anomaly Detection"])
def get_historical_anomalies(asset: str, top_n: int = 20, threshold: float = 60.0):
    """
    Historical anomaly events (market crashes).

    Returns top anomaly events clustered into 5-day windows,
    matched with known crash events from crash_labels.json.

    Parameters:
    - top_n: Number of events to return (default: 20)
    - threshold: Minimum anomaly score (default: 60.0)
    """
    a = _check_asset(asset)
    try:
        return historical_anomalies(a, top_n=top_n, threshold=threshold)
    except Exception as e:
        log.error(f"historical_anomalies error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/comparison/{asset}", tags=["Anomaly Detection"])
def get_anomaly_model_comparison(asset: str):
    """
    Per-model anomaly detection statistics.

    Returns performance stats and correlation matrix for all 4
    anomaly detection models (Z-Score, IForest, LSTM, Prophet).
    """
    a = _check_asset(asset)
    try:
        return model_comparison(a)
    except Exception as e:
        log.error(f"model_comparison error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/evaluation", tags=["Anomaly Detection"])
def get_anomaly_evaluation():
    """
    Full anomaly detection evaluation report.

    Returns Precision/Recall/F1/AUC metrics for all models across
    all assets, evaluated against 24 labeled crash events.
    """
    report_path = ROOT_DIR / "backend" / "models" / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404,
                            detail="Evaluation report not found. Run evaluate.py first.")
    with open(report_path) as f:
        return json.load(f)


@app.get("/anomaly/crash-labels", tags=["Anomaly Detection"])
def get_crash_labels():
    """Return labeled crash events used for objective validation overlays."""
    try:
        return load_crash_labels()
    except Exception as e:
        log.error(f"crash-labels error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/metrics/{asset}", tags=["Anomaly Detection"])
def get_anomaly_metrics(
    asset: str,
    model: str = "ensemble_score",
    threshold: float = 60.0,
    window_days: int = 7,
):
    """Return AUCPR/Brier/calibration/lead-time metrics for anomaly scoring."""
    a = _check_asset(asset)
    if not 0 <= threshold <= 100:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 100")
    if not 1 <= window_days <= 30:
        raise HTTPException(status_code=400, detail="window_days must be between 1 and 30")

    try:
        return anomaly_metrics(a, model=model, threshold=threshold, window_days=window_days)
    except Exception as e:
        log.error(f"metrics error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/threshold-analysis/{asset}", tags=["Anomaly Detection"])
def get_threshold_analysis(
    asset: str,
    model: str = "ensemble_score",
    min_threshold: float = 40.0,
    max_threshold: float = 80.0,
    step: float = 2.0,
    cost_fp: float = 1.0,
    cost_fn: float = 5.0,
    window_days: int = 7,
):
    """Sweep thresholds and return precision/recall/F1 plus utility-optimal threshold."""
    a = _check_asset(asset)
    if not 0 <= min_threshold < max_threshold <= 100:
        raise HTTPException(status_code=400, detail="threshold range must satisfy 0 <= min < max <= 100")
    if step <= 0:
        raise HTTPException(status_code=400, detail="step must be > 0")
    if cost_fp < 0 or cost_fn < 0:
        raise HTTPException(status_code=400, detail="cost_fp and cost_fn must be >= 0")

    try:
        return anomaly_threshold_analysis(
            a,
            model=model,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            step=step,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            window_days=window_days,
        )
    except Exception as e:
        log.error(f"threshold-analysis error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/false-positives/{asset}", tags=["Anomaly Detection"])
def get_false_positives(
    asset: str,
    model: str = "ensemble_score",
    threshold: float = 60.0,
    top_n: int = 40,
    window_days: int = 7,
):
    """Return high-score dates not matched to labeled crash windows."""
    a = _check_asset(asset)
    if not 0 <= threshold <= 100:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 100")
    if not 1 <= top_n <= 500:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 500")

    try:
        return false_positive_timeline(
            a,
            model=model,
            threshold=threshold,
            top_n=top_n,
            window_days=window_days,
        )
    except Exception as e:
        log.error(f"false-positives error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/bubble-risk/{asset}", tags=["Anomaly Detection"])
def get_bubble_risk(asset: str):
    """Return a bubble-risk snapshot derived from engineered valuation and stress signals."""
    a = _check_asset(asset)
    try:
        return bubble_risk_snapshot(a)
    except Exception as e:
        log.error(f"bubble-risk error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: ADVANCED MODEL ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/anomaly/advanced/{asset}", tags=["Anomaly Detection - Advanced"])
def get_advanced_anomaly(asset: str):
    """
    Advanced anomaly analysis with canonical 9-model output.

    Returns:
    - 9 model score keys (4 baseline + 5 advanced)
      Missing model values are returned as null and listed in missing_models
    - Baseline ensemble vs Advanced ensemble
    - Current market regime (bull/bear/crisis)
    - Risk assessment using advanced models

    Phase 2 addition: XGBoost (supervised), HMM (regime), TCN,
    VAE, and Anomaly Transformer.
    """
    a = _check_asset(asset)
    try:
        return advanced_current_analysis(a)
    except Exception as e:
        log.error(f"advanced_current_analysis error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/regime/{asset}", tags=["Anomaly Detection - Advanced"])
def get_regime_timeline(asset: str):
    """
    HMM market regime timeline and statistics.

    Returns:
    - Full historical regime timeline (bull/bear/crisis)
    - Regime transition matrix
    - Average returns per regime
    - Current regime state

    Useful for understanding market cycles and risk periods.
    """
    a = _check_asset(asset)
    try:
        return regime_timeline(a)
    except Exception as e:
        log.error(f"regime_timeline error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/anomaly/compare-tiers/{asset}", tags=["Anomaly Detection - Advanced"])
def compare_model_tiers(asset: str):
    """
    Compare baseline (4 models) vs advanced (9-model contract) ensemble.

    Returns side-by-side comparison:
    - Current baseline ensemble score
    - Current advanced ensemble score
    - Improvement delta
    - Historical performance comparison

    Shows the value of Phase 2 advanced models.
    """
    a = _check_asset(asset)
    try:
        baseline = current_analysis(a)
        advanced = advanced_current_analysis(a)

        return {
            "asset": a,
            "date": baseline["date"],
            "baseline": {
                "ensemble_score": baseline["ensemble_score"],
                "risk_label": baseline["risk_label"],
                "models": baseline["model_scores"],
            },
            "advanced": {
                "ensemble_score": advanced["advanced_ensemble"],
                "risk_label": advanced["risk_label"],
                "regime": advanced["current_regime"],
                "models": advanced["model_scores"],
            },
            "improvement": {
                "score_delta": round(advanced["advanced_ensemble"] - baseline["ensemble_score"], 2),
                "better_detection": advanced["advanced_ensemble"] > baseline["ensemble_score"],
            },
        }
    except Exception as e:
        log.error(f"compare_tiers error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3A: DEEP LEARNING FORECASTING (NEW)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/forecast/lstm/{asset}", tags=["Forecasting - Deep Learning"])
def forecast_lstm(asset: str, horizon: int = 30):
    """
    LSTM Seq2Seq price forecast with attention weights.

    Returns:
    - Point forecast (median prediction)
    - Lower/upper 95% confidence intervals
    - Attention weights showing which past values influenced predictions
    """
    a = _check_asset(asset)
    try:
        result = lstm_seq2seq_forecast(a, horizon=horizon)
        # Get current price
        features_data = load_all_features()
        current_price = float(features_data[a]["Close"].iloc[-1]) if a in features_data else result["forecast"][0]
        forecast_last = result["forecast"][-1] if result["forecast"] else current_price
        return {
            **result,
            "current_price": current_price,
            "summary": {
                "forecast_30d": float(forecast_last),
                "expected_return_pct": round(((float(forecast_last) / current_price) - 1) * 100, 2) if current_price else 0,
            },
        }
    except Exception as e:
        log.error(f"LSTM forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/transformer/{asset}", tags=["Forecasting - Deep Learning"])
def forecast_transformer_endpoint(asset: str, horizon: int = 30):
    """
    Transformer-based price forecast with multi-head self-attention.

    Returns:
    - Point forecast
    - Lower/upper 95% confidence intervals
    - Transformer provides attention over historical context
    """
    a = _check_asset(asset)
    try:
        result = transformer_forecast(a, horizon=horizon)
        features_data = load_all_features()
        current_price = float(features_data[a]["Close"].iloc[-1]) if a in features_data else result["forecast"][0]
        forecast_last = result["forecast"][-1] if result["forecast"] else current_price
        return {
            **result,
            "current_price": current_price,
            "summary": {
                "forecast_30d": float(forecast_last),
                "expected_return_pct": round(((float(forecast_last) / current_price) - 1) * 100, 2) if current_price else 0,
            },
        }
    except Exception as e:
        log.error(f"Transformer forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/xgboost-price/{asset}", tags=["Forecasting - Deep Learning"])
def forecast_xgboost_endpoint(asset: str, horizon: int = 30):
    """
    XGBoost gradient boosting price forecast with feature importance.

    Returns:
    - Point forecast
    - Lower/upper 95% confidence intervals (from quantile regression)
    - SHAP feature importance showing which features drive predictions
    """
    a = _check_asset(asset)
    try:
        result = xgboost_forecast(a, horizon=horizon)
        features_data = load_all_features()
        current_price = float(features_data[a]["Close"].iloc[-1]) if a in features_data else result["forecast"][0]
        forecast_last = result["forecast"][-1] if result["forecast"] else current_price
        return {
            **result,
            "current_price": current_price,
            "summary": {
                "forecast_30d": float(forecast_last),
                "expected_return_pct": round(((float(forecast_last) / current_price) - 1) * 100, 2) if current_price else 0,
            },
        }
    except Exception as e:
        log.error(f"XGBoost forecast error for {a}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/scores/{asset}")
async def websocket_live_scores(websocket: WebSocket, asset: str):
    """
    WebSocket endpoint for real-time anomaly score updates.
    Sends updated scores every 60 seconds.
    """
    await websocket.accept()
    a = asset.upper()
    if a not in ASSETS:
        await websocket.close(code=4004)
        return

    try:
        while True:
            # Get current scores from predict module
            data = current_analysis(a)
            adv = advanced_current_analysis(a)
            payload = {
                "asset": a,
                "timestamp": pd.Timestamp.now().isoformat(),
                "baseline_score": data["ensemble_score"],
                "advanced_score": adv["advanced_ensemble"],
                "regime": adv["current_regime"],
                "risk_label": adv["risk_label"],
                "model_scores": {**data["model_scores"], **{
                    "xgb": adv["model_scores"]["xgb"],
                    "hmm": adv["model_scores"]["hmm"],
                    "tcn": adv["model_scores"]["tcn"],
                }},
            }
            await websocket.send_json(payload)
            await asyncio.sleep(60)  # refresh every 60 seconds
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WebSocket error for {a}: {e}")
        await websocket.close(code=1011)

