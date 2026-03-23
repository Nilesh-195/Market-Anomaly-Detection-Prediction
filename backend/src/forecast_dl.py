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

from dl_models import forecast_dl_model
from gb_models import forecast_xgboost_recursive
from features import load_all_features

logger = logging.getLogger(__name__)

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

    model = torch.load(model_path, map_location='cpu')
    model.eval()

    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        meta = None

    return model, meta


def _load_xgboost_model(asset):
    """Load XGBoost models.

    Args:
        asset: Asset name

    Returns:
        models: Dict with 3 quantile models
        scaler: sklearn MinMaxScaler
        feature_names: List of feature names
    """
    meta_path = MODEL_DIR / asset / "xgboost_meta.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(f"XGBoost meta not found: {meta_path}")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    return meta.get('models'), meta.get('scaler'), meta.get('feature_names')


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

        # Get last 30 days normalized
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

        # Get last 30 days normalized
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
    """Generate XGBoost forecast with feature importance.

    Args:
        asset: Asset name
        horizon: Prediction horizon

    Returns:
        dict: {
            "asset": str,
            "date": str,
            "method": "xgboost",
            "forecast": [float] × horizon,
            "lower_95": [float] × horizon,
            "upper_95": [float] × horizon,
            "dates": [str] × horizon,
            "feature_importance": {feature_name: float},
            "model_info": {
                "type": "gradient boosting (XGBoost)",
                "n_estimators": 100
            }
        }
    """
    try:
        # Load models and data
        models, scaler, feature_names = _load_xgboost_model(asset)
        features_data = load_all_features()
        df = features_data[asset]

        # Get last prices + features (normalized)
        last_prices = df['Close'].iloc[-30:].values
        last_prices_norm = scaler.transform(last_prices.reshape(-1, 1)).flatten()

        # Get current features (limit to feature_names count - 30 for lags)
        n_feature_cols = len(feature_names) - 30
        if n_feature_cols > 0:
            last_features = df.iloc[-1, :n_feature_cols].values
            # Normalize features (simple z-score)
            last_features = (last_features - last_features.mean()) / (last_features.std() + 1e-8)
        else:
            last_features = np.array([])

        # Forecast
        point, lower, upper, importance = forecast_xgboost_recursive(
            models, scaler, last_prices_norm, last_features, horizon, feature_names
        )

        # Generate dates
        last_date = pd.Timestamp(df.index[-1])
        dates = [
            (last_date + pd.Timedelta(days=i + 1)).strftime('%Y-%m-%d')
            for i in range(len(point))
        ]

        # Convert importance dict to JSON-serializable format
        importance_dict = {}
        if importance:
            for k, v in importance.items():
                importance_dict[str(k)] = float(v) if hasattr(v, 'item') else float(v)

        return {
            "asset": asset,
            "date": str(df.index[-1].date()),
            "method": "xgboost",
            "forecast": [float(x) for x in point.tolist()[:horizon]],
            "lower_95": [float(x) for x in lower.tolist()[:horizon]],
            "upper_95": [float(x) for x in upper.tolist()[:horizon]],
            "dates": dates[:horizon],
            "feature_importance": importance_dict,
            "model_info": {
                "type": "gradient boosting (XGBoost)",
                "n_estimators": 100,
                "quantiles": [0.025, 0.5, 0.975]
            }
        }

    except Exception as e:
        logger.error(f"XGBoost forecast error for {asset}: {e}")
        raise
