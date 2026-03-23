"""
Forecasting Functions for DL Models
====================================
High-level interface for LSTM, Transformer, and XGBoost forecasting

Provides:
  - lstm_seq2seq_forecast()
  - transformer_forecast()
  - xgboost_forecast()

Each returns forecast, confidence intervals, and explainability metrics
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import logging

from dl_models import forecast_dl_model, LSTMSeq2Seq, TransformerForecaster
from gb_models import forecast_xgboost_recursive
from features import load_all_features

logger = logging.getLogger(__name__)

# Allowlist custom model classes so torch.load works with weights_only=True as well
try:
    torch.serialization.add_safe_globals([LSTMSeq2Seq, TransformerForecaster])
except AttributeError:
    pass  # Older torch versions don't have add_safe_globals

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models"


def _load_dl_model(asset, model_name):
    """Load LSTM/Transformer model.

    Args:
        asset: Asset name (e.g., 'SP500')
        model_name: 'lstm_seq2seq' or 'transformer'

    Returns:
        model: PyTorch model
        meta: Scaler info {"min": float, "max": float}
    """
    model_path = MODEL_DIR / asset / f"{model_name}.pt"
    meta_path = MODEL_DIR / asset / f"{model_name}_meta.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # weights_only=False needed to load full model objects (PyTorch 2.6+)
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()

    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        meta = None

    return model, meta


def _load_xgboost_model(asset):
    """Load XGBoost models. Returns full meta dict."""
    meta_path = MODEL_DIR / asset / "xgboost_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"XGBoost meta not found: {meta_path}")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta


def lstm_seq2seq_forecast(asset, horizon=30):
    """Generate LSTM Seq2Seq forecast with attention weights.

    Args:
        asset: Asset name (e.g., 'SP500')
        horizon: Prediction horizon (days)

    Returns:
        dict: {
            "asset": str,
            "date": str,
            "method": "lstm_seq2seq",
            "forecast": [float] × horizon,
            "lower_95": [float] × horizon,
            "upper_95": [float] × horizon,
            "dates": [str] × horizon,
            "attention_weights": [[float]] × horizon × lookback,
            "model_info": {
                "lookback": int,
                "quantiles": [0.025, 0.5, 0.975]
            }
        }
    """
    try:
        # Load model and data
        model, scaler = _load_dl_model(asset, 'lstm_seq2seq')
        features_data = load_all_features()
        df = features_data[asset]

        mode = scaler.get('mode', 'prices')
        current_price = float(df['Close'].iloc[-1])

        if mode == 'returns':
            # New returns-based mode
            last_prices = df['Close'].iloc[-31:].values
            returns = np.log(last_prices[1:] / last_prices[:-1])
            mean_val = scaler['mean']
            std_val = scaler['std']
            normalized = (returns - mean_val) / (std_val + 1e-8)
        else:
            # Legacy absolute prices mode
            min_val = scaler['min']
            max_val = scaler['max']
            range_val = max_val - min_val + 1e-8
            last_prices = df['Close'].iloc[-30:].values
            normalized = (last_prices - min_val) / range_val

        # Forecast
        with torch.no_grad():
            X = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(-1)
            output, attention = model(X)

            # Extract quantiles
            output = output.numpy().squeeze(0)
            point = output[:, 1]
            lower = output[:, 0]
            upper = output[:, 2]

            if mode == 'returns':
                point_ret = point * std_val + mean_val
                lower_ret = lower * std_val + mean_val
                upper_ret = upper * std_val + mean_val

                point = current_price * np.exp(np.cumsum(point_ret))
                lower = current_price * np.exp(np.cumsum(lower_ret))
                upper = current_price * np.exp(np.cumsum(upper_ret))
            else:
                # Denormalize
                point = point * range_val + min_val
                lower = lower * range_val + min_val
                upper = upper * range_val + min_val

            # Attention weights
            attn_weights = (
                attention.numpy().squeeze(0) if attention is not None else None
            )

        # Generate dates
        last_date = pd.Timestamp(df.index[-1])
        dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d')
            for i in range(len(point))
        ]

        return {
            "asset": asset,
            "date": str(df.index[-1].date()),
            "method": "lstm_seq2seq",
            "forecast": point.tolist()[:horizon],
            "lower_95": lower.tolist()[:horizon],
            "upper_95": upper.tolist()[:horizon],
            "dates": dates[:horizon],
            "attention_weights": (
                attn_weights[:horizon].tolist() if attn_weights is not None else None
            ),
            "model_info": {
                "lookback": 30,
                "quantiles": [0.025, 0.5, 0.975],
                "type": "encoder-decoder LSTM with attention"
            }
        }

    except Exception as e:
        logger.error(f"LSTM forecast error for {asset}: {e}")
        raise


def transformer_forecast(asset, horizon=30):
    """Generate Transformer forecast with attention heatmap.

    Args:
        asset: Asset name
        horizon: Prediction horizon

    Returns:
        dict: Same structure as LSTM but with transformer-specific attention
    """
    try:
        # Load model and data
        model, scaler = _load_dl_model(asset, 'transformer')
        features_data = load_all_features()
        df = features_data[asset]

        mode = scaler.get('mode', 'prices')
        current_price = float(df['Close'].iloc[-1])

        if mode == 'returns':
            # New returns-based mode
            last_prices = df['Close'].iloc[-31:].values
            returns = np.log(last_prices[1:] / last_prices[:-1])
            mean_val = scaler['mean']
            std_val = scaler['std']
            normalized = (returns - mean_val) / (std_val + 1e-8)
        else:
            # Legacy absolute prices mode
            min_val = scaler['min']
            max_val = scaler['max']
            range_val = max_val - min_val + 1e-8
            last_prices = df['Close'].iloc[-30:].values
            normalized = (last_prices - min_val) / range_val

        # Forecast
        with torch.no_grad():
            X = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(-1)
            output, _ = model(X)

            # Extract quantiles
            output = output.numpy().squeeze(0)
            point = output[:, 1]
            lower = output[:, 0]
            upper = output[:, 2]

            if mode == 'returns':
                point_ret = point * std_val + mean_val
                lower_ret = lower * std_val + mean_val
                upper_ret = upper * std_val + mean_val

                point = current_price * np.exp(np.cumsum(point_ret))
                lower = current_price * np.exp(np.cumsum(lower_ret))
                upper = current_price * np.exp(np.cumsum(upper_ret))
            else:
                # Denormalize
                point = point * range_val + min_val
                lower = lower * range_val + min_val
                upper = upper * range_val + min_val

        # Generate dates
        last_date = pd.Timestamp(df.index[-1])
        dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d')
            for i in range(len(point))
        ]

        return {
            "asset": asset,
            "date": str(df.index[-1].date()),
            "method": "transformer",
            "forecast": point.tolist()[:horizon],
            "lower_95": lower.tolist()[:horizon],
            "upper_95": upper.tolist()[:horizon],
            "dates": dates[:horizon],
            "attention_weights": None,  # Would require model architecture changes
            "model_info": {
                "lookback": 30,
                "quantiles": [0.025, 0.5, 0.975],
                "type": "Transformer with multi-head self-attention"
            }
        }

    except Exception as e:
        logger.error(f"Transformer forecast error for {asset}: {e}")
        raise


def xgboost_forecast(asset, horizon=30):
    """Generate XGBoost forecast using log-return compounding.

    The model predicts next-day log-return from the last 30 daily returns.
    Returns are compounded recursively for multi-step forecasting.
    This avoids the extrapolation failure of absolute-price XGBoost models.
    """
    try:
        meta = _load_xgboost_model(asset)
        models = meta.get('models')
        scaler = meta.get('scaler')
        feature_names = meta.get('feature_names', [])
        importance_dict = meta.get('feature_importance', {})
        mode = meta.get('mode', 'prices')   # 'returns' for new models

        features_data = load_all_features()
        df = features_data[asset]
        close = df['Close'].values

        if mode == 'returns':
            # ── Returns-based forecast (new) ──────────────────────────
            log_returns = np.log(close[1:] / close[:-1])  # (n-1,)
            last_returns = log_returns[-30:]               # last 30 returns
            current_price = float(close[-1])

            forecasts_point, forecasts_lower, forecasts_upper = [], [], []

            for step in range(horizon):
                X = last_returns.reshape(1, -1)
                X_scaled = scaler.transform(X)

                r_point = float(models[0.5].predict(X_scaled)[0])
                r_lower = float(models[0.025].predict(X_scaled)[0])
                r_upper = float(models[0.975].predict(X_scaled)[0])

                prev = forecasts_point[-1] if forecasts_point else current_price
                forecasts_point.append(prev * np.exp(r_point))
                forecasts_lower.append(prev * np.exp(r_lower))
                forecasts_upper.append(prev * np.exp(r_upper))

                # Advance return window
                last_returns = np.roll(last_returns, -1)
                last_returns[-1] = r_point

        else:
            # ── Absolute-price fallback (legacy) ──────────────────────
            last_prices = close[-30:]
            last_prices_norm = scaler.transform(last_prices.reshape(-1, 1)).flatten()
            last_features = np.array([])

            from gb_models import forecast_xgboost_recursive
            point, lower, upper, importance_dict = forecast_xgboost_recursive(
                models, scaler, last_prices_norm, last_features, horizon, feature_names
            )
            forecasts_point = point.tolist()
            forecasts_lower = lower.tolist()
            forecasts_upper = upper.tolist()
            current_price = float(close[-1])

        # Generate future dates (skip weekends)
        last_date = pd.Timestamp(df.index[-1])
        dates = []
        d = last_date
        while len(dates) < horizon:
            d += pd.Timedelta(days=1)
            if d.weekday() < 5:  # Mon–Fri
                dates.append(d.strftime('%Y-%m-%d'))

        # Convert importance to JSON-safe dict
        if importance_dict:
            importance_dict = {
                str(k): float(v) if hasattr(v, 'item') else float(v)
                for k, v in importance_dict.items()
            }

        return {
            "asset": asset,
            "date": str(df.index[-1].date()),
            "method": "xgboost",
            "forecast": [float(x) for x in forecasts_point[:horizon]],
            "lower_95": [float(x) for x in forecasts_lower[:horizon]],
            "upper_95": [float(x) for x in forecasts_upper[:horizon]],
            "dates": dates[:horizon],
            "feature_importance": importance_dict,
            "model_info": {
                "type": "gradient boosting (XGBoost, log-return mode)",
                "n_estimators": 300,
                "quantiles": [0.025, 0.5, 0.975],
                "mode": mode,
            }
        }

    except Exception as e:
        logger.error(f"XGBoost forecast error for {asset}: {e}")
        raise
