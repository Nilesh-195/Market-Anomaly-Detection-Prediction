"""
forecast_evaluation.py
======================
Comprehensive forecast evaluation module for TSFA course project.

Implements:
  1. Forecast Accuracy Metrics
     - RMSE (Root Mean Squared Error)
     - MAE (Mean Absolute Error)
     - MAPE (Mean Absolute Percentage Error)
     - SMAPE (Symmetric MAPE)
     - MASE (Mean Absolute Scaled Error)
     - Theil's U statistic

  2. Cross-Validation Methods
     - Rolling window (sliding window)
     - Expanding window
     - Time series k-fold

  3. Prediction Interval Evaluation
     - Coverage probability
     - Mean interval width
     - Winkler score

  4. Method Comparison
     - Side-by-side comparison tables
     - Statistical significance tests (Diebold-Mariano)
     - Ranking by multiple criteria

  5. Report Generation
     - CSV exports
     - Visualization plots
     - Summary statistics

Theory:
  Time series cross-validation differs from regular CV because:
  - Cannot shuffle data (temporal ordering matters)
  - Must train on past, test on future
  - Rolling/expanding windows simulate real forecasting
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "backend" / "data" / "processed"
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FORECAST ACCURACY METRICS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ForecastMetrics:
    """Container for all forecast accuracy metrics."""
    rmse: float
    mae: float
    mape: float
    smape: float
    mase: float
    theils_u: float
    me: float  # Mean Error (bias)
    mpe: float  # Mean Percentage Error

    def to_dict(self) -> dict:
        return {
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE": self.mape,
            "SMAPE": self.smape,
            "MASE": self.mase,
            "Theil_U": self.theils_u,
            "ME": self.me,
            "MPE": self.mpe,
        }


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt(mean((actual - predicted)²))

    Properties:
    - Same units as the data
    - Penalizes large errors more than MAE
    - Most common metric for regression
    """
    errors = actual - predicted
    return np.sqrt(np.mean(errors ** 2))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = mean(|actual - predicted|)

    Properties:
    - Same units as the data
    - More robust to outliers than RMSE
    - Easier to interpret
    """
    return np.mean(np.abs(actual - predicted))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    MAPE = mean(|actual - predicted| / |actual|) × 100

    Properties:
    - Percentage scale (0-100+)
    - Scale-independent
    - WARNING: Undefined when actual = 0
    """
    # Avoid division by zero
    nonzero_mask = actual != 0
    if nonzero_mask.sum() == 0:
        return np.nan

    percentage_errors = np.abs(actual[nonzero_mask] - predicted[nonzero_mask]) / np.abs(actual[nonzero_mask])
    return np.mean(percentage_errors) * 100


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.

    SMAPE = mean(|actual - predicted| / ((|actual| + |predicted|) / 2)) × 100

    Properties:
    - Bounded between 0% and 200%
    - Symmetric (treats over/under-prediction equally)
    - More stable than MAPE
    """
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2

    nonzero_mask = denominator != 0
    if nonzero_mask.sum() == 0:
        return np.nan

    return np.mean(numerator[nonzero_mask] / denominator[nonzero_mask]) * 100


def calculate_mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: np.ndarray,
    seasonal_period: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE / MAE_naive

    Where MAE_naive is from seasonal naive forecast on training data.

    Properties:
    - Scale-independent
    - Compares to naive benchmark
    - < 1 means better than naive, > 1 means worse
    """
    # Calculate naive forecast errors on training data
    if seasonal_period == 1:
        naive_errors = np.abs(np.diff(train))
    else:
        naive_errors = np.abs(train[seasonal_period:] - train[:-seasonal_period])

    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.nan

    mae_forecast = calculate_mae(actual, predicted)
    return mae_forecast / mae_naive


def calculate_theils_u(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Theil's U Statistic (U2).

    Compares forecast to naive (no-change) forecast.

    U2 = sqrt(sum((predicted - actual)²)) / sqrt(sum((actual_{t+1} - actual_t)²))

    Properties:
    - U2 < 1: Better than naive
    - U2 = 1: Same as naive
    - U2 > 1: Worse than naive
    """
    if len(actual) < 2:
        return np.nan

    # Naive forecast: y_{t+1} = y_t
    naive_pred = np.roll(actual, 1)[1:]
    actual_shifted = actual[1:]
    predicted_shifted = predicted[1:]

    numerator = np.sqrt(np.sum((predicted_shifted - actual_shifted) ** 2))
    denominator = np.sqrt(np.sum((actual_shifted - naive_pred) ** 2))

    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_me(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Error (Bias).

    ME = mean(actual - predicted)

    Properties:
    - Positive ME: Model under-predicts on average
    - Negative ME: Model over-predicts on average
    - Good forecasts have ME ≈ 0
    """
    return np.mean(actual - predicted)


def calculate_mpe(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Mean Percentage Error.

    MPE = mean((actual - predicted) / actual) × 100

    Properties:
    - Shows directional bias as percentage
    - Positive = under-prediction, Negative = over-prediction
    """
    nonzero_mask = actual != 0
    if nonzero_mask.sum() == 0:
        return np.nan

    percentage_errors = (actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask]
    return np.mean(percentage_errors) * 100


def calculate_all_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train: Optional[np.ndarray] = None,
    seasonal_period: int = 1
) -> ForecastMetrics:
    """
    Calculate all forecast accuracy metrics.

    Parameters
    ----------
    actual : np.ndarray
        True values
    predicted : np.ndarray
        Forecasted values
    train : np.ndarray, optional
        Training data (needed for MASE)
    seasonal_period : int
        Seasonal period for MASE calculation

    Returns
    -------
    ForecastMetrics dataclass with all metrics
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    # Remove NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    if len(actual) == 0:
        return ForecastMetrics(
            rmse=np.nan, mae=np.nan, mape=np.nan, smape=np.nan,
            mase=np.nan, theils_u=np.nan, me=np.nan, mpe=np.nan
        )

    # MASE needs training data
    if train is not None:
        train = np.array(train).flatten()
        mase = calculate_mase(actual, predicted, train, seasonal_period)
    else:
        mase = np.nan

    return ForecastMetrics(
        rmse=calculate_rmse(actual, predicted),
        mae=calculate_mae(actual, predicted),
        mape=calculate_mape(actual, predicted),
        smape=calculate_smape(actual, predicted),
        mase=mase,
        theils_u=calculate_theils_u(actual, predicted),
        me=calculate_me(actual, predicted),
        mpe=calculate_mpe(actual, predicted),
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CROSS-VALIDATION METHODS
# ══════════════════════════════════════════════════════════════════════════════

def rolling_window_cv(
    series: pd.Series,
    forecast_func: Callable,
    initial_train_size: int = 252,
    horizon: int = 1,
    step: int = 1,
    **forecast_kwargs
) -> dict:
    """
    Rolling (sliding) window cross-validation.

    The training window slides forward, maintaining constant size.

    Example (initial=100, horizon=5, step=5):
      Fold 1: Train [0:100],   Test [100:105]
      Fold 2: Train [5:105],   Test [105:110]
      Fold 3: Train [10:110],  Test [110:115]
      ...

    Parameters
    ----------
    series : pd.Series
        Full time series
    forecast_func : Callable
        Function that takes (train, horizon) and returns forecast
    initial_train_size : int
        Size of training window (fixed)
    horizon : int
        Forecast horizon
    step : int
        Step size between folds

    Returns
    -------
    dict with:
        - all_actuals: List of actual values per fold
        - all_forecasts: List of forecasts per fold
        - metrics_per_fold: DataFrame with metrics for each fold
        - aggregate_metrics: Overall metrics
    """
    n = len(series)
    all_actuals = []
    all_forecasts = []
    fold_metrics = []

    fold = 0
    start = 0

    while start + initial_train_size + horizon <= n:
        train_end = start + initial_train_size
        test_end = train_end + horizon

        train = series.iloc[start:train_end]
        test = series.iloc[train_end:test_end]

        try:
            forecast = forecast_func(train, horizon, **forecast_kwargs)

            if hasattr(forecast, "values"):
                forecast_values = forecast.values[:len(test)]
            else:
                forecast_values = np.array(forecast)[:len(test)]

            all_actuals.extend(test.values)
            all_forecasts.extend(forecast_values)

            # Calculate fold metrics
            metrics = calculate_all_metrics(test.values, forecast_values, train.values)
            fold_metrics.append({
                "Fold": fold,
                "Train_Start": train.index[0],
                "Train_End": train.index[-1],
                "Test_Start": test.index[0],
                "Test_End": test.index[-1],
                **metrics.to_dict(),
            })

        except Exception as e:
            log.warning(f"Fold {fold} failed: {e}")

        fold += 1
        start += step

    # Aggregate metrics
    fold_df = pd.DataFrame(fold_metrics)
    aggregate = calculate_all_metrics(
        np.array(all_actuals),
        np.array(all_forecasts),
        series.values[:initial_train_size]
    )

    log.info(f"Rolling CV: {fold} folds, Aggregate RMSE={aggregate.rmse:.4f}")

    return {
        "all_actuals": np.array(all_actuals),
        "all_forecasts": np.array(all_forecasts),
        "metrics_per_fold": fold_df,
        "aggregate_metrics": aggregate,
        "n_folds": fold,
    }


def expanding_window_cv(
    series: pd.Series,
    forecast_func: Callable,
    initial_train_size: int = 252,
    horizon: int = 1,
    step: int = 1,
    **forecast_kwargs
) -> dict:
    """
    Expanding window cross-validation.

    The training window grows over time (uses all available history).

    Example (initial=100, horizon=5, step=5):
      Fold 1: Train [0:100],   Test [100:105]
      Fold 2: Train [0:105],   Test [105:110]  ← Training expands
      Fold 3: Train [0:110],   Test [110:115]
      ...

    This is more common in practice as it uses all available data.

    Parameters
    ----------
    series : pd.Series
        Full time series
    forecast_func : Callable
        Function that takes (train, horizon) and returns forecast
    initial_train_size : int
        Minimum training size
    horizon : int
        Forecast horizon
    step : int
        Step size between folds

    Returns
    -------
    dict with folds, metrics, and aggregate results
    """
    n = len(series)
    all_actuals = []
    all_forecasts = []
    fold_metrics = []

    fold = 0
    train_end = initial_train_size

    while train_end + horizon <= n:
        test_end = train_end + horizon

        train = series.iloc[:train_end]  # Expanding: always start from 0
        test = series.iloc[train_end:test_end]

        try:
            forecast = forecast_func(train, horizon, **forecast_kwargs)

            if hasattr(forecast, "values"):
                forecast_values = forecast.values[:len(test)]
            else:
                forecast_values = np.array(forecast)[:len(test)]

            all_actuals.extend(test.values)
            all_forecasts.extend(forecast_values)

            metrics = calculate_all_metrics(test.values, forecast_values, train.values)
            fold_metrics.append({
                "Fold": fold,
                "Train_Size": len(train),
                "Train_End": train.index[-1],
                "Test_Start": test.index[0],
                "Test_End": test.index[-1],
                **metrics.to_dict(),
            })

        except Exception as e:
            log.warning(f"Fold {fold} failed: {e}")

        fold += 1
        train_end += step

    fold_df = pd.DataFrame(fold_metrics)
    aggregate = calculate_all_metrics(
        np.array(all_actuals),
        np.array(all_forecasts),
        series.values[:initial_train_size]
    )

    log.info(f"Expanding CV: {fold} folds, Aggregate RMSE={aggregate.rmse:.4f}")

    return {
        "all_actuals": np.array(all_actuals),
        "all_forecasts": np.array(all_forecasts),
        "metrics_per_fold": fold_df,
        "aggregate_metrics": aggregate,
        "n_folds": fold,
    }


def time_series_kfold_cv(
    series: pd.Series,
    forecast_func: Callable,
    n_splits: int = 5,
    horizon: int = 30,
    **forecast_kwargs
) -> dict:
    """
    Time series k-fold cross-validation.

    Divides data into k roughly equal folds, training on all previous folds.

    Example (n_splits=5):
      Fold 1: Train [0:20%],    Test [20%:40%]
      Fold 2: Train [0:40%],    Test [40%:60%]
      Fold 3: Train [0:60%],    Test [60%:80%]
      Fold 4: Train [0:80%],    Test [80%:100%]

    Parameters
    ----------
    series : pd.Series
        Full time series
    forecast_func : Callable
        Forecasting function
    n_splits : int
        Number of splits
    horizon : int
        Forecast horizon (used within each fold)

    Returns
    -------
    dict with fold results and aggregate metrics
    """
    n = len(series)
    fold_size = n // (n_splits + 1)

    all_actuals = []
    all_forecasts = []
    fold_metrics = []

    for fold in range(1, n_splits + 1):
        train_end = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)

        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]

        if len(test) == 0:
            continue

        try:
            # Use min of horizon and available test data
            actual_horizon = min(horizon, len(test))
            forecast = forecast_func(train, actual_horizon, **forecast_kwargs)

            if hasattr(forecast, "values"):
                forecast_values = forecast.values[:actual_horizon]
            else:
                forecast_values = np.array(forecast)[:actual_horizon]

            test_values = test.values[:actual_horizon]

            all_actuals.extend(test_values)
            all_forecasts.extend(forecast_values)

            metrics = calculate_all_metrics(test_values, forecast_values, train.values)
            fold_metrics.append({
                "Fold": fold,
                "Train_Size": len(train),
                "Test_Size": len(test_values),
                **metrics.to_dict(),
            })

        except Exception as e:
            log.warning(f"Fold {fold} failed: {e}")

    fold_df = pd.DataFrame(fold_metrics)
    aggregate = calculate_all_metrics(
        np.array(all_actuals),
        np.array(all_forecasts)
    )

    log.info(f"TS K-Fold: {n_splits} folds, Aggregate RMSE={aggregate.rmse:.4f}")

    return {
        "all_actuals": np.array(all_actuals),
        "all_forecasts": np.array(all_forecasts),
        "metrics_per_fold": fold_df,
        "aggregate_metrics": aggregate,
        "n_folds": len(fold_metrics),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PREDICTION INTERVAL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def calculate_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """
    Calculate prediction interval coverage probability.

    Coverage = (# of actual values within interval) / (total predictions)

    For a 95% prediction interval, coverage should be ≈ 95%.
    - Under-coverage (< 95%): Intervals too narrow
    - Over-coverage (> 95%): Intervals too wide (conservative)

    Parameters
    ----------
    actual : np.ndarray
        True values
    lower : np.ndarray
        Lower prediction interval bounds
    upper : np.ndarray
        Upper prediction interval bounds

    Returns
    -------
    float: Coverage probability (0-1)
    """
    within_interval = (actual >= lower) & (actual <= upper)
    return np.mean(within_interval)


def calculate_mean_interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
    actual: Optional[np.ndarray] = None,
    normalize: bool = True
) -> float:
    """
    Calculate mean prediction interval width.

    Width = mean(upper - lower)

    If normalize=True, returns width as percentage of actual values.

    Parameters
    ----------
    lower : np.ndarray
        Lower bounds
    upper : np.ndarray
        Upper bounds
    actual : np.ndarray, optional
        Actual values (for normalization)
    normalize : bool
        Whether to normalize by actual values

    Returns
    -------
    float: Mean interval width (raw or percentage)
    """
    width = upper - lower

    if normalize and actual is not None:
        nonzero = actual != 0
        if nonzero.sum() > 0:
            return np.mean(width[nonzero] / np.abs(actual[nonzero])) * 100

    return np.mean(width)


def calculate_winkler_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05
) -> float:
    """
    Calculate Winkler score for prediction intervals.

    Winkler score penalizes both:
    1. Wide intervals (reduced sharpness)
    2. Actual values outside intervals (coverage failure)

    S_t = (U_t - L_t) + (2/α)(L_t - Y_t)·I(Y_t < L_t) + (2/α)(Y_t - U_t)·I(Y_t > U_t)

    Lower is better.

    Parameters
    ----------
    actual : np.ndarray
        True values
    lower : np.ndarray
        Lower bounds
    upper : np.ndarray
        Upper bounds
    alpha : float
        Significance level (0.05 for 95% interval)

    Returns
    -------
    float: Mean Winkler score
    """
    width = upper - lower

    # Penalty for values below lower bound
    below_penalty = (2 / alpha) * (lower - actual) * (actual < lower)

    # Penalty for values above upper bound
    above_penalty = (2 / alpha) * (actual - upper) * (actual > upper)

    scores = width + below_penalty + above_penalty
    return np.mean(scores)


def evaluate_prediction_intervals(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    confidence: float = 0.95,
    name: str = "Model"
) -> dict:
    """
    Comprehensive evaluation of prediction intervals.

    Parameters
    ----------
    actual : np.ndarray
        True values
    lower : np.ndarray
        Lower prediction interval bounds
    upper : np.ndarray
        Upper prediction interval bounds
    confidence : float
        Nominal confidence level
    name : str
        Model name for logging

    Returns
    -------
    dict with all interval metrics
    """
    actual = np.array(actual).flatten()
    lower = np.array(lower).flatten()
    upper = np.array(upper).flatten()

    # Remove NaN
    mask = ~(np.isnan(actual) | np.isnan(lower) | np.isnan(upper))
    actual, lower, upper = actual[mask], lower[mask], upper[mask]

    coverage = calculate_coverage(actual, lower, upper)
    mean_width = calculate_mean_interval_width(lower, upper, actual, normalize=True)
    winkler = calculate_winkler_score(actual, lower, upper, alpha=1 - confidence)

    # Check if coverage is within acceptable range
    # For 95% CI, acceptable coverage is roughly 90-98%
    target = confidence
    is_well_calibrated = abs(coverage - target) < 0.05

    results = {
        "coverage": coverage,
        "target_coverage": target,
        "mean_width_pct": mean_width,
        "winkler_score": winkler,
        "is_calibrated": is_well_calibrated,
        "n_predictions": len(actual),
    }

    status = "✓ Well-calibrated" if is_well_calibrated else "✗ Mis-calibrated"
    log.info(f"[{name}] PI Evaluation: Coverage={coverage:.1%} (target={target:.0%}), "
             f"Width={mean_width:.1f}%, {status}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: STATISTICAL COMPARISON TESTS
# ══════════════════════════════════════════════════════════════════════════════

def diebold_mariano_test(
    actual: np.ndarray,
    forecast1: np.ndarray,
    forecast2: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided"
) -> dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t)
    L is a loss function (default: squared error)

    Parameters
    ----------
    actual : np.ndarray
        True values
    forecast1 : np.ndarray
        First model's forecasts
    forecast2 : np.ndarray
        Second model's forecasts
    h : int
        Forecast horizon (for HAC standard errors)
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    dict with:
        - dm_statistic: DM test statistic
        - p_value: p-value
        - conclusion: Which model is better (if significant)
    """
    actual = np.array(actual).flatten()
    forecast1 = np.array(forecast1).flatten()
    forecast2 = np.array(forecast2).flatten()

    # Calculate loss differential (squared errors)
    e1 = actual - forecast1
    e2 = actual - forecast2
    d = e1**2 - e2**2

    n = len(d)
    d_mean = np.mean(d)

    # HAC variance estimation (Newey-West)
    # Simplified: use autocorrelation up to h lags
    gamma_0 = np.var(d)
    gamma_sum = 0
    for k in range(1, h + 1):
        if k < n:
            gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            gamma_sum += 2 * (1 - k / (h + 1)) * gamma_k

    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        var_d = gamma_0 / n

    dm_stat = d_mean / np.sqrt(var_d)

    # p-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif alternative == "less":
        p_value = stats.norm.cdf(dm_stat)
    else:  # greater
        p_value = 1 - stats.norm.cdf(dm_stat)

    # Conclusion
    if p_value < 0.05:
        if dm_stat > 0:
            conclusion = "Model 2 is significantly better"
        else:
            conclusion = "Model 1 is significantly better"
    else:
        conclusion = "No significant difference"

    return {
        "dm_statistic": dm_stat,
        "p_value": p_value,
        "conclusion": conclusion,
        "mean_loss_diff": d_mean,
    }


def compare_models_statistical(
    actual: np.ndarray,
    forecasts: dict[str, np.ndarray],
    baseline: str = "naive"
) -> pd.DataFrame:
    """
    Compare multiple models against a baseline using DM test.

    Parameters
    ----------
    actual : np.ndarray
        True values
    forecasts : dict
        Dictionary of model_name -> forecast array
    baseline : str
        Name of baseline model to compare against

    Returns
    -------
    pd.DataFrame with pairwise comparison results
    """
    if baseline not in forecasts:
        log.warning(f"Baseline '{baseline}' not found in forecasts")
        return pd.DataFrame()

    baseline_forecast = forecasts[baseline]
    results = []

    for model_name, model_forecast in forecasts.items():
        if model_name == baseline:
            continue

        dm_result = diebold_mariano_test(actual, baseline_forecast, model_forecast)

        results.append({
            "Model": model_name,
            "vs_Baseline": baseline,
            "DM_Statistic": dm_result["dm_statistic"],
            "p_value": dm_result["p_value"],
            "Significant": dm_result["p_value"] < 0.05,
            "Conclusion": dm_result["conclusion"],
        })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: METHOD COMPARISON TABLE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_comparison_table(
    results: dict[str, dict],
    metrics: list[str] = ["RMSE", "MAE", "MAPE", "MASE"]
) -> pd.DataFrame:
    """
    Generate side-by-side comparison table of all methods.

    Parameters
    ----------
    results : dict
        Dictionary of method_name -> {metrics: ..., ...}
    metrics : list
        Metrics to include in comparison

    Returns
    -------
    pd.DataFrame with methods as rows, metrics as columns
    """
    rows = []

    for method_name, method_results in results.items():
        row = {"Method": method_name}

        if "metrics" in method_results:
            method_metrics = method_results["metrics"]
            if isinstance(method_metrics, ForecastMetrics):
                method_metrics = method_metrics.to_dict()
        elif "aggregate_metrics" in method_results:
            method_metrics = method_results["aggregate_metrics"]
            if isinstance(method_metrics, ForecastMetrics):
                method_metrics = method_metrics.to_dict()
        else:
            continue

        for metric in metrics:
            row[metric] = method_metrics.get(metric, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by RMSE (or first metric)
    if "RMSE" in df.columns:
        df = df.sort_values("RMSE")

    # Add rank
    df["Rank"] = range(1, len(df) + 1)

    return df


def create_heatmap_comparison(
    comparison_df: pd.DataFrame,
    asset_name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create heatmap visualization of method comparison.
    """
    metrics = [col for col in comparison_df.columns if col not in ["Method", "Rank"]]

    # Normalize each metric to 0-1 scale (lower is better)
    normalized = comparison_df[metrics].copy()
    for col in metrics:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(normalized.values, cmap="RdYlGn_r", aspect="auto")

    # Labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(comparison_df)))
    ax.set_yticklabels(comparison_df["Method"])

    # Colorbar
    plt.colorbar(im, ax=ax, label="Normalized Score (lower = better)")

    # Add value annotations
    for i in range(len(comparison_df)):
        for j, metric in enumerate(metrics):
            value = comparison_df.iloc[i][metric]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(f"{asset_name} - Method Comparison Heatmap")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved heatmap → {save_path.name}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_cv_results(
    cv_result: dict,
    name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot cross-validation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} - Cross-Validation Results", fontsize=14, fontweight="bold")

    fold_df = cv_result["metrics_per_fold"]

    # RMSE by fold
    ax = axes[0, 0]
    ax.plot(fold_df["Fold"], fold_df["RMSE"], marker="o")
    ax.axhline(cv_result["aggregate_metrics"].rmse, color="red", linestyle="--",
               label=f"Aggregate: {cv_result['aggregate_metrics'].rmse:.2f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Fold")
    ax.legend()

    # MAE by fold
    ax = axes[0, 1]
    ax.plot(fold_df["Fold"], fold_df["MAE"], marker="o", color="green")
    ax.axhline(cv_result["aggregate_metrics"].mae, color="red", linestyle="--",
               label=f"Aggregate: {cv_result['aggregate_metrics'].mae:.2f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Fold")
    ax.legend()

    # Actual vs Forecast scatter
    ax = axes[1, 0]
    ax.scatter(cv_result["all_actuals"], cv_result["all_forecasts"], alpha=0.5, s=10)
    max_val = max(cv_result["all_actuals"].max(), cv_result["all_forecasts"].max())
    min_val = min(cv_result["all_actuals"].min(), cv_result["all_forecasts"].min())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect forecast")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Forecast")
    ax.set_title("Actual vs Forecast")
    ax.legend()

    # Error distribution
    ax = axes[1, 1]
    errors = cv_result["all_actuals"] - cv_result["all_forecasts"]
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("Forecast Error")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution (Mean: {np.mean(errors):.2f})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved CV plot → {save_path.name}")

    return fig


def plot_comparison_bar(
    comparison_df: pd.DataFrame,
    metric: str = "RMSE",
    asset_name: str = "Asset",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create bar chart comparison of methods.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = comparison_df["Method"]
    values = comparison_df[metric]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(methods)))
    bars = ax.barh(methods, values, color=colors)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01 * max(values), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)

    ax.set_xlabel(metric)
    ax.set_title(f"{asset_name} - {metric} Comparison (lower is better)")
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved comparison bar → {save_path.name}")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all_methods(
    series: pd.Series,
    name: str,
    test_size: int = 30,
    cv_folds: int = 5,
    save_plots: bool = True
) -> dict:
    """
    Comprehensive evaluation of all forecasting methods for one asset.

    Parameters
    ----------
    series : pd.Series
        Full price series
    name : str
        Asset name
    test_size : int
        Hold-out test size
    cv_folds : int
        Number of cross-validation folds
    save_plots : bool
        Whether to save plots

    Returns
    -------
    dict with all evaluation results
    """
    from naive_methods import naive_forecast, mean_forecast, drift_forecast
    from exponential_smoothing import ses_forecast, holt_forecast
    from arima_models import arima_forecast

    log.info(f"\n{'='*60}")
    log.info(f"[{name}] COMPREHENSIVE FORECAST EVALUATION")
    log.info(f"{'='*60}")

    # Train/test split
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    horizon = len(test)

    # Define forecast functions
    def naive_func(tr, h):
        return naive_forecast(tr, h)["forecast"]

    def mean_func(tr, h):
        return mean_forecast(tr, h)["forecast"]

    def drift_func(tr, h):
        return drift_forecast(tr, h)["forecast"]

    def ses_func(tr, h):
        return ses_forecast(tr, h)["forecast"]

    def holt_func(tr, h):
        return holt_forecast(tr, h)["forecast"]

    def arima_func(tr, h):
        result = arima_forecast(tr, h, order=(1, 1, 1), name=name)
        return result["forecast"]

    methods = {
        "Naive": naive_func,
        "Mean": mean_func,
        "Drift": drift_func,
        "SES": ses_func,
        "Holt": holt_func,
        "ARIMA(1,1,1)": arima_func,
    }

    results = {}

    # 1. Simple hold-out evaluation
    log.info(f"[{name}] Evaluating on hold-out test set (n={test_size})...")
    holdout_results = {}

    for method_name, forecast_func in methods.items():
        try:
            forecast = forecast_func(train, horizon)
            forecast_values = forecast.values if hasattr(forecast, "values") else np.array(forecast)

            metrics = calculate_all_metrics(test.values, forecast_values, train.values)
            holdout_results[method_name] = {
                "forecast": forecast_values,
                "metrics": metrics,
            }
            log.info(f"  {method_name}: RMSE={metrics.rmse:.2f}, MAE={metrics.mae:.2f}")

        except Exception as e:
            log.warning(f"  {method_name} failed: {e}")

    results["holdout"] = holdout_results

    # 2. Cross-validation (expanding window)
    log.info(f"[{name}] Running expanding window CV...")
    cv_results = {}

    for method_name, forecast_func in methods.items():
        try:
            cv_result = expanding_window_cv(
                series,
                forecast_func,
                initial_train_size=len(train) - cv_folds * test_size,
                horizon=test_size,
                step=test_size,
            )
            cv_results[method_name] = cv_result

        except Exception as e:
            log.warning(f"  CV {method_name} failed: {e}")

    results["cv"] = cv_results

    # 3. Generate comparison table
    comparison = generate_comparison_table(holdout_results)
    results["comparison"] = comparison

    # 4. Statistical tests (DM test vs naive)
    if "Naive" in holdout_results and len(holdout_results) > 1:
        forecasts = {k: v["forecast"] for k, v in holdout_results.items()}
        dm_comparison = compare_models_statistical(test.values, forecasts, baseline="Naive")
        results["dm_tests"] = dm_comparison
        log.info(f"[{name}] Diebold-Mariano tests completed")

    # 5. Find best method
    if not comparison.empty:
        best_method = comparison.iloc[0]["Method"]
        best_rmse = comparison.iloc[0]["RMSE"]
        results["best_method"] = best_method
        results["best_rmse"] = best_rmse
        log.info(f"[{name}] Best method: {best_method} (RMSE={best_rmse:.2f})")

    # 6. Save plots
    if save_plots:
        # Comparison bar chart
        bar_path = RESULTS_DIR / f"{name}_comparison_bar.png"
        plot_comparison_bar(comparison, "RMSE", name, bar_path)
        plt.close()

        # CV results for best method
        if best_method in cv_results:
            cv_path = RESULTS_DIR / f"{name}_cv_results.png"
            plot_cv_results(cv_results[best_method], f"{name} - {best_method}", cv_path)
            plt.close()

        # Heatmap
        heatmap_path = RESULTS_DIR / f"{name}_heatmap.png"
        create_heatmap_comparison(comparison, name, heatmap_path)
        plt.close()

    # Summary
    summary = f"""
    Evaluation Summary: {name}
    ─────────────────────────────
    Test size: {test_size} observations
    Methods evaluated: {len(holdout_results)}

    Hold-out Results (by RMSE):
{comparison.to_string(index=False)}

    Best Method: {results.get('best_method', 'N/A')}

    Cross-Validation Summary:
    {f"Best CV RMSE: {cv_results.get(best_method, {}).get('aggregate_metrics', ForecastMetrics(np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)).rmse:.2f}" if best_method in cv_results else 'N/A'}
    """
    log.info(summary)
    results["summary"] = summary

    return results


def evaluate_all_assets(
    test_size: int = 30,
    save_plots: bool = True
) -> dict:
    """
    Run comprehensive evaluation on all 6 assets.
    """
    log.info("=" * 70)
    log.info("COMPREHENSIVE FORECAST EVALUATION - ALL ASSETS")
    log.info("=" * 70)

    all_results = {}
    summary_rows = []

    for asset in ASSETS:
        path = PROCESSED_DIR / f"{asset}_features.parquet"
        if not path.exists():
            log.warning(f"[{asset}] Not found — skipping")
            continue

        df = pd.read_parquet(path)
        series = df["Close"]

        results = evaluate_all_methods(series, asset, test_size, save_plots=save_plots)
        all_results[asset] = results

        # Add to summary
        if "comparison" in results and not results["comparison"].empty:
            for _, row in results["comparison"].iterrows():
                summary_rows.append({
                    "Asset": asset,
                    "Method": row["Method"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                    "Rank": row["Rank"],
                })

    # Save master summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "evaluation_master_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Master summary saved → {summary_path.name}")

    # Best method per asset
    best_by_asset = summary_df[summary_df["Rank"] == 1][["Asset", "Method", "RMSE"]]
    best_path = RESULTS_DIR / "best_methods_by_asset.csv"
    best_by_asset.to_csv(best_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("BEST FORECASTING METHOD BY ASSET")
    print("=" * 70)
    print(best_by_asset.to_string(index=False))
    print("=" * 70)

    # Overall winner frequency
    method_counts = summary_df[summary_df["Rank"] == 1]["Method"].value_counts()
    print("\nMETHOD WINS (how often each is best):")
    print(method_counts.to_string())
    print("=" * 70)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Forecast Evaluation Module")
    parser.add_argument("--test-size", type=int, default=30, help="Test set size")
    parser.add_argument("--asset", type=str, help="Single asset to evaluate")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot saving")

    args = parser.parse_args()

    if args.asset:
        # Single asset evaluation
        path = PROCESSED_DIR / f"{args.asset}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            results = evaluate_all_methods(
                df["Close"],
                args.asset,
                args.test_size,
                save_plots=not args.no_plots
            )
        else:
            log.error(f"Asset {args.asset} not found")
    else:
        # All assets
        results = evaluate_all_assets(
            test_size=args.test_size,
            save_plots=not args.no_plots
        )
