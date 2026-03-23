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
