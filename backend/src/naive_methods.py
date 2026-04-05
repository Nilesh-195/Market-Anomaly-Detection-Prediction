"""
naive_methods.py
================
Baseline forecasting methods for time series prediction.

Implements 4 naive benchmark methods as taught in TSFA course:

  1. Mean Method
     - Forecast = mean(training data)
     - Assumes series fluctuates around constant mean
     - Best for: stationary series with no trend/seasonality

  2. Naive (Random Walk) Method
     - Forecast = last observed value
     - ŷ_{T+h} = y_T for all h
     - Best for: random walk series, efficient markets

  3. Seasonal Naive Method
     - Forecast = value from same season last period
     - ŷ_{T+h} = y_{T+h-m} where m = seasonal period
     - Best for: strongly seasonal series

  4. Drift Method
     - Forecast follows line from first to last observation
     - ŷ_{T+h} = y_T + h × (y_T - y_1) / (T - 1)
     - Best for: series with clear linear trend

Theory:
  These naive methods serve as baselines. If a sophisticated model
  (ARIMA, exponential smoothing) can't beat these benchmarks,
  the model adds no value.

  For stock prices (random walk hypothesis), the Naive method
  is often hard to beat, especially for short horizons.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features import ASSETS

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
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "naive_forecasts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Evaluation Metrics ─────────────────────────────────────────────────────────
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate forecast accuracy metrics.

    Parameters
    ----------
    actual : np.ndarray
        True values
    predicted : np.ndarray
        Forecasted values

    Returns
    -------
    dict with RMSE, MAE, MAPE, SMAPE
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Remove any NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    if len(actual) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "smape": np.nan}

    errors = actual - predicted
    abs_errors = np.abs(errors)

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)

    # MAPE (exclude zeros in actual to avoid division by zero)
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(abs_errors[nonzero_mask] / np.abs(actual[nonzero_mask])) * 100
    else:
        mape = np.nan

    # SMAPE (Symmetric MAPE)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    nonzero_denom = denominator != 0
    if nonzero_denom.sum() > 0:
        smape = np.mean(abs_errors[nonzero_denom] / denominator[nonzero_denom]) * 100
    else:
        smape = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
    }


# ── Method 1: Mean Forecast ────────────────────────────────────────────────────
def mean_forecast(
    train: pd.Series,
    horizon: int,
    confidence: float = 0.95
) -> dict:
    """
    Mean method: forecast is the average of training data.

    ŷ_{T+h} = mean(y_1, ..., y_T) for all h

    Parameters
    ----------
    train : pd.Series
        Training data (historical observations)
    horizon : int
        Number of steps to forecast
    confidence : float
        Confidence level for prediction intervals

    Returns
    -------
    dict with:
        - forecast: pd.Series of point forecasts
        - lower: Lower prediction interval
        - upper: Upper prediction interval
        - method: 'mean'
    """
    mean_value = train.mean()
    std_value = train.std()

    # Create forecast index
    last_date = train.index[-1]
    freq = pd.infer_freq(train.index) or "D"
    forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    # Point forecast
    forecast = pd.Series([mean_value] * horizon, index=forecast_index)

    # Prediction intervals: widen as horizon increases
    # SE = sqrt(1 + 1/T) × σ for h=1, grows with sqrt(h)
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    n = len(train)

    se = std_value * np.sqrt(1 + 1/n + np.arange(1, horizon + 1) / n)
    lower = forecast - z * se
    upper = forecast + z * se

    log.info(f"Mean method: forecast={mean_value:.2f}, std={std_value:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "method": "mean",
        "mean_value": mean_value,
    }


# ── Method 2: Naive (Random Walk) Forecast ─────────────────────────────────────
def naive_forecast(
    train: pd.Series,
    horizon: int,
    confidence: float = 0.95
) -> dict:
    """
    Naive method: forecast is the last observed value.

    ŷ_{T+h} = y_T for all h

    This is equivalent to a random walk model.

    Parameters
    ----------
    train : pd.Series
        Training data
    horizon : int
        Forecast horizon
    confidence : float
        Confidence level

    Returns
    -------
    dict with forecast, lower, upper bounds
    """
    last_value = train.iloc[-1]

    # Create forecast index
    last_date = train.index[-1]
    freq = pd.infer_freq(train.index) or "D"
    forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    # Point forecast
    forecast = pd.Series([last_value] * horizon, index=forecast_index)

    # Prediction intervals: based on random walk variance
    # For random walk, variance at h steps = h × σ²
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    # Estimate σ from returns
    returns = train.diff().dropna()
    sigma = returns.std()

    # SE grows with sqrt(h) for random walk
    se = sigma * np.sqrt(np.arange(1, horizon + 1))
    lower = forecast - z * se
    upper = forecast + z * se

    log.info(f"Naive method: last_value={last_value:.2f}, sigma={sigma:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "method": "naive",
        "last_value": last_value,
    }


# ── Method 3: Seasonal Naive Forecast ──────────────────────────────────────────
def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int,
    seasonal_period: int = 21,  # ~1 month for daily trading data
    confidence: float = 0.95
) -> dict:
    """
    Seasonal naive method: forecast equals value from same season last period.

    ŷ_{T+h} = y_{T+h-km}  where k is chosen so T+h-km is the most recent
    observation from the same season.

    Parameters
    ----------
    train : pd.Series
        Training data
    horizon : int
        Forecast horizon
    seasonal_period : int
        Number of periods in one season (default 21 = monthly for trading days)
    confidence : float
        Confidence level

    Returns
    -------
    dict with forecast, lower, upper bounds
    """
    m = seasonal_period

    # Create forecast index
    last_date = train.index[-1]
    freq = pd.infer_freq(train.index) or "D"
    forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    # Build forecasts by looking back m periods
    forecasts = []
    for h in range(1, horizon + 1):
        # Look back to same seasonal position
        lookback_idx = len(train) - m + ((h - 1) % m)
        if lookback_idx >= 0 and lookback_idx < len(train):
            forecasts.append(train.iloc[lookback_idx])
        else:
            # Not enough history, fall back to last value
            forecasts.append(train.iloc[-1])

    forecast = pd.Series(forecasts, index=forecast_index)

    # Prediction intervals
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    # Estimate seasonal variance
    seasonal_diffs = train.diff(m).dropna()
    sigma = seasonal_diffs.std() if len(seasonal_diffs) > 0 else train.std()

    # Number of complete seasons in each forecast
    k = (np.arange(1, horizon + 1) - 1) // m + 1
    se = sigma * np.sqrt(k)
    lower = forecast - z * se
    upper = forecast + z * se

    log.info(f"Seasonal Naive: period={m}, sigma={sigma:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "method": "seasonal_naive",
        "seasonal_period": m,
    }


# ── Method 4: Drift Forecast ───────────────────────────────────────────────────
def drift_forecast(
    train: pd.Series,
    horizon: int,
    confidence: float = 0.95
) -> dict:
    """
    Drift method: forecast follows a line from first to last observation.

    ŷ_{T+h} = y_T + h × (y_T - y_1) / (T - 1)

    Equivalent to a random walk with drift (average change).

    Parameters
    ----------
    train : pd.Series
        Training data
    horizon : int
        Forecast horizon
    confidence : float
        Confidence level

    Returns
    -------
    dict with forecast, lower, upper bounds
    """
    last_value = train.iloc[-1]
    first_value = train.iloc[0]
    n = len(train)

    # Average drift (slope)
    drift = (last_value - first_value) / (n - 1)

    # Create forecast index
    last_date = train.index[-1]
    freq = pd.infer_freq(train.index) or "D"
    forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

    # Point forecast
    h_values = np.arange(1, horizon + 1)
    forecast_values = last_value + drift * h_values
    forecast = pd.Series(forecast_values, index=forecast_index)

    # Prediction intervals
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    # Estimate residual variance around drift line
    fitted = first_value + drift * np.arange(n)
    residuals = train.values - fitted
    sigma = np.std(residuals)

    # SE for drift method
    se = sigma * np.sqrt(h_values * (1 + h_values / n))
    lower = forecast - z * se
    upper = forecast + z * se

    log.info(f"Drift method: drift={drift:.4f}/day, start={first_value:.2f}, end={last_value:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "method": "drift",
        "drift": drift,
    }


# ── Run All Naive Methods ──────────────────────────────────────────────────────
def run_all_naive_methods(
    train: pd.Series,
    test: pd.Series,
    seasonal_period: int = 21,
    name: str = "Series",
    horizon: Optional[int] = None
) -> dict:
    """
    Run all 4 naive methods and compare performance.

    Parameters
    ----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data (for evaluation)
    seasonal_period : int
        Seasonal period for seasonal naive
    name : str
        Name for logging

    Returns
    -------
    dict with results for each method and comparison table
    """
    horizon = horizon if horizon is not None else len(test)
    log.info(f"\n[{name}] Running naive methods — horizon={horizon}")

    # Run each method
    methods = {
        "mean": mean_forecast(train, horizon),
        "naive": naive_forecast(train, horizon),
        "seasonal_naive": seasonal_naive_forecast(train, horizon, seasonal_period),
        "drift": drift_forecast(train, horizon),
    }

    comparison = []
    for method_name, result in methods.items():
        if len(test) > 0:
            metrics = calculate_metrics(test.values, result["forecast"].values)
            result["metrics"] = metrics
        else:
            metrics = {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "smape": np.nan}
            
        comparison.append({
            "Method": method_name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "MAPE": metrics["mape"],
            "SMAPE": metrics["smape"],
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values("RMSE")
    comparison_df["Rank"] = range(1, len(comparison_df) + 1)

    best_method = comparison_df.iloc[0]["Method"]
    log.info(f"[{name}] Best naive method: {best_method} (RMSE={comparison_df.iloc[0]['RMSE']:.2f})")

    return {
        "methods": methods,
        "comparison": comparison_df,
        "best_method": best_method,
        "test_series": test,
        "train_series": train,
    }


# ── Visualization ──────────────────────────────────────────────────────────────
def plot_naive_forecasts(
    results: dict,
    name: str,
    show_last_n: int = 100,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot all naive method forecasts with actual values.
    """
    train = results["train_series"]
    test = results["test_series"]
    methods = results["methods"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} - Naive Forecasting Methods Comparison", fontsize=14, fontweight="bold")

    method_names = ["mean", "naive", "seasonal_naive", "drift"]
    colors = ["blue", "green", "orange", "red"]

    for idx, (method_name, color) in enumerate(zip(method_names, colors)):
        ax = axes[idx // 2, idx % 2]
        result = methods[method_name]

        # Plot training data (last n points)
        train_plot = train.iloc[-show_last_n:]
        ax.plot(train_plot.index, train_plot.values, label="Training", color="gray", alpha=0.7)

        # Plot actual test values
        ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)

        # Plot forecast
        ax.plot(result["forecast"].index, result["forecast"].values,
                label=f"Forecast ({method_name})", color=color, linewidth=2, linestyle="--")

        # Plot confidence interval
        ax.fill_between(result["forecast"].index,
                        result["lower"].values,
                        result["upper"].values,
                        alpha=0.2, color=color, label="95% CI")

        # Add metrics
        metrics = result["metrics"]
        ax.set_title(f"{method_name.replace('_', ' ').title()}\n"
                    f"RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved plot → {save_path.name}")

    return fig


# ── Main Analysis Function ─────────────────────────────────────────────────────
def analyze_naive_methods(
    series: pd.Series,
    name: str,
    test_size: int = 30,
    seasonal_period: int = 21,
    save_plots: bool = True
) -> dict:
    """
    Complete naive methods analysis for an asset.

    Parameters
    ----------
    series : pd.Series
        Full price series
    name : str
        Asset name
    test_size : int
        Number of observations to hold out for testing
    seasonal_period : int
        Seasonal period for seasonal naive method
    save_plots : bool
        Whether to save plots

    Returns
    -------
    dict with all results and comparison
    """
    log.info(f"\n{'='*60}")
    log.info(f"[{name}] Naive Methods Analysis")
    log.info(f"{'='*60}")

    # Train/test split
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    log.info(f"[{name}] Train: {len(train)} obs ({train.index[0].date()} → {train.index[-1].date()})")
    log.info(f"[{name}] Test:  {len(test)} obs ({test.index[0].date()} → {test.index[-1].date()})")

    # Run all methods
    results = run_all_naive_methods(train, test, seasonal_period, name)

    # Generate summary
    summary = f"""
    Naive Methods Analysis: {name}
    ─────────────────────────────
    Train size: {len(train)} observations
    Test size:  {len(test)} observations

    Method Comparison:
{results['comparison'].to_string(index=False)}

    Best Method: {results['best_method']}
    """
    log.info(summary)

    # Save plot
    if save_plots:
        plot_path = RESULTS_DIR / f"{name}_naive_forecasts.png"
        plot_naive_forecasts(results, name, save_path=plot_path)
        plt.close()

    results["summary"] = summary
    return results


def analyze_all_assets(test_size: int = 30) -> dict:
    """
    Run naive methods analysis on all 6 assets.
    """
    log.info("=" * 60)
    log.info("Naive Methods Analysis - All Assets")
    log.info("=" * 60)

    results = {}
    summary_rows = []

    for name in ASSETS:
        path = PROCESSED_DIR / f"{name}_features.parquet"
        if not path.exists():
            log.warning(f"[{name}] Features not found — skipping.")
            continue

        df = pd.read_parquet(path)
        series = df["Close"]

        result = analyze_naive_methods(series, name, test_size=test_size)
        results[name] = result

        # Add to summary
        for _, row in result["comparison"].iterrows():
            summary_rows.append({
                "Asset": name,
                "Method": row["Method"],
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "MAPE": row["MAPE"],
                "Rank": row["Rank"],
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "naive_methods_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path.name}")

    # Create pivot for best method by asset
    best_by_asset = summary_df[summary_df["Rank"] == 1][["Asset", "Method", "RMSE"]]
    print("\n" + "=" * 60)
    print("NAIVE METHODS - BEST METHOD BY ASSET")
    print("=" * 60)
    print(best_by_asset.to_string(index=False))
    print("=" * 60)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = analyze_all_assets(test_size=30)
