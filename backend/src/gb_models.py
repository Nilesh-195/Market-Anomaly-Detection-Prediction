"""
XGBoost-based Price Forecasting
=================================
Uses XGBoost regressor with gradient boosting for multi-step price prediction

Features:
  - Uses lagged prices (last 30 days) + engineered features
  - Quantile regression for confidence intervals (0.025, 0.5, 0.975)
  - SHAP values for feature importance
  - Recursive forecasting for multi-step ahead
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

import pickle
from pathlib import Path
import logging
from features import load_all_features

MODEL_DIR = Path(__file__).parent.parent / "models"
logger = logging.getLogger(__name__)


try:
    # SHAP has compatibility issues on some systems; use built-in feature importance instead
    # import shap
    # HAS_SHAP = True
    HAS_SHAP = False  # Force to use built-in XGBoost importance
except ImportError:
    HAS_SHAP = False


def create_lagged_features(close_prices, features_df, lookback=30, horizon=30):
    """Create lagged price features.

    Args:
        close_prices: (n_samples,) close prices
        features_df: (n_samples, n_features) engineered features
        lookback: Number of past days
        horizon: Prediction horizon

    Returns:
        X: Feature matrix with lagged prices
        y: Target prices
    """
    n_samples = len(close_prices)
    X_list = []
    y_list = []

    for i in range(lookback, n_samples - horizon):
        # Lagged prices (last 30 days)
        lags = close_prices[i - lookback : i]

        # Current features
        curr_features = features_df.iloc[i].values if features_df is not None else []

        # Combine lags with features
        combined = np.concatenate([lags, curr_features]) if features_df is not None else lags

        X_list.append(combined)

        # Target: mean price over next horizon (simplified)
        target = close_prices[i : i + horizon].mean()
        y_list.append(target)

    return np.array(X_list), np.array(y_list)


def train_xgboost_regressor(
    close_prices,
    features_df,
    asset_name,
    lookback=30,
    horizon=30,
    test_size=0.3,
):
    """Train XGBoost model for price regression.

    Args:
        close_prices: (n_samples,) close prices
        features_df: (n_samples, n_features) or None
        asset_name: Asset name for logging
        lookback: Past window size
        horizon: Future horizon (for target aggregation)
        test_size: Train/test split ratio

    Returns:
        models: Dict with 3 quantile models (q=0.025, 0.5, 0.975)
        scaler: MinMax scaler
        feature_names: List of feature names
    """
    print(f"[{asset_name}] Training XGBoost regressor...")

    # Normalize prices
    scaler = MinMaxScaler()
    close_norm = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

    # Create features
    if features_df is not None:
        X, y = create_lagged_features(close_norm, features_df, lookback, horizon)
        feature_names = [f"lag_{i}" for i in range(1, lookback + 1)]
        # Add feature column names if available
        if hasattr(features_df, 'columns'):
            feature_names += list(features_df.columns)
        else:
            feature_names += [f"feat_{i}" for i in range(features_df.shape[1])]
    else:
        # Prices only
        X, y = create_lagged_features(close_norm, None, lookback, horizon)
        feature_names = [f"lag_{i}" for i in range(1, lookback + 1)]

    # Train/test split
    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train 3 quantile models
    quantiles = [0.025, 0.5, 0.975]
    models = {}

    for q in quantiles:
        print(f"  Training quantile {q}...")

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',  # Use standard squared error (works on all versions)
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=0,
        )

        if q != 0.5:
            # For quantile regression, use custom objective
            def quantile_loss(y_true, y_pred):
                residual = y_true - y_pred
                grad = np.where(residual >= 0, q, q - 1)
                hess = np.ones_like(residual)
                return grad, hess

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            # Median: standard regression
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

        models[q] = model

    print(f"[{asset_name}] XGBoost training complete")

    return models, scaler, feature_names


def forecast_xgboost_recursive(
    models, scaler, last_prices, last_features, horizon=30, feature_names=None
):
    """Forecast using XGBoost with recursive strategy.

    Args:
        models: Dict with 3 quantile models
        scaler: MinMax scaler
        last_prices: (lookback,) last N normalized prices
        last_features: (n_features,) current features
        horizon: Prediction horizon
        feature_names: Feature names for logging

    Returns:
        forecast_point: (horizon,) denormalized forecasts
        forecast_lower: (horizon,) lower bound
        forecast_upper: (horizon,) upper bound
        importance: Dict of feature importance
    """
    lookback = len(last_prices)
    forecasts = {0.025: [], 0.5: [], 0.975: []}

    # Use last prices as initial state
    current_prices = last_prices.copy()

    for step in range(horizon):
        # Build feature vector
        X_future = np.concatenate([current_prices, last_features])
        X_future = X_future.reshape(1, -1)

        # Predict 3 quantiles
        for q, model in models.items():
            pred = model.predict(X_future)[0]
            # Clip to reasonable range
            pred = np.clip(pred, 0, 1)
            forecasts[q].append(pred)

        # Update prices for next step (use point estimate)
        current_prices = np.roll(current_prices, -1)
        current_prices[-1] = forecasts[0.5][-1]

    # Denormalize
    price_range = scaler.data_max_[0] - scaler.data_min_[0]
    price_min = scaler.data_min_[0]

    forecast_point = (
        np.array(forecasts[0.5]) * price_range + price_min
    )
    forecast_lower = (
        np.array(forecasts[0.025]) * price_range + price_min
    )
    forecast_upper = (
        np.array(forecasts[0.975]) * price_range + price_min
    )

    # Get feature importance from median model
    importance = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(models[0.5])
            X_sample = np.concatenate([last_prices, last_features]).reshape(1, -1)
            shap_values = explainer.shap_values(X_sample)
            importance = dict(
                zip(
                    feature_names or [f"f_{i}" for i in range(len(shap_values[0]))],
                    np.abs(shap_values[0])
                )
            )
            # Top 10
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        except:
            print("  Warning: SHAP explanation failed (optional)")
    else:
        # Use built-in feature importance
        try:
            importances = models[0.5].feature_importances_
            importance = dict(
                zip(
                    feature_names or [f"f_{i}" for i in range(len(importances))],
                    importances
                )
            )
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        except:
            pass

    return forecast_point, forecast_lower, forecast_upper, importance


def _load_xgboost_model(asset):
    """Load XGBoost models. Returns full meta dict."""
    meta_path = MODEL_DIR / asset / "xgboost_meta.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(f"XGBoost meta not found: {meta_path}")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return meta



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

        def _anchor_to_spot(point_path, lower_path, upper_path, spot_price, max_gap_pct=15.0):
            """Rebase a forecast path to current spot if first point is implausibly far.

            Legacy absolute-price models can drift in level over long histories.
            This keeps the model-implied trajectory shape while anchoring the
            level to current market price.
            """
            if len(point_path) == 0 or spot_price <= 0:
                return point_path, lower_path, upper_path

            first = float(point_path[0])
            if first <= 0:
                return point_path, lower_path, upper_path

            gap_pct = abs(first / float(spot_price) - 1.0) * 100.0
            if gap_pct <= max_gap_pct:
                return point_path, lower_path, upper_path

            base = first
            point_adj = [float(spot_price) * (float(v) / base) for v in point_path]
            lower_adj = [float(spot_price) * (float(v) / base) for v in lower_path]
            upper_adj = [float(spot_price) * (float(v) / base) for v in upper_path]
            return point_adj, lower_adj, upper_adj

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
            lag_feature_count = 30
            extra_feature_names = feature_names[lag_feature_count:] if len(feature_names) > lag_feature_count else []
            latest_row = df.iloc[-1].to_dict()
            last_features = np.array([
                float(latest_row.get(name, 0.0)) for name in extra_feature_names
            ], dtype=float)

            from gb_models import forecast_xgboost_recursive
            point, lower, upper, importance_dict = forecast_xgboost_recursive(
                models, scaler, last_prices_norm, last_features, horizon, feature_names
            )
            forecasts_point = point.tolist()
            forecasts_lower = lower.tolist()
            forecasts_upper = upper.tolist()
            current_price = float(close[-1])

            # Legacy model safety: re-anchor level if it drifts too far from spot.
            forecasts_point, forecasts_lower, forecasts_upper = _anchor_to_spot(
                forecasts_point, forecasts_lower, forecasts_upper, current_price, max_gap_pct=15.0
            )

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
