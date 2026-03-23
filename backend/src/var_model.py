"""
var_model.py
============
Vector Autoregression (VAR) for multi-asset time series forecasting.

VAR models capture interdependencies between multiple time series,
making them ideal for forecasting correlated financial assets.

Key Concepts:
  - VAR(p): Each variable is a linear function of p lags of ALL variables
  - Granger Causality: Tests if past values of X help predict Y
  - Impulse Response Function (IRF): Effect of a shock in one variable on others
  - Forecast Error Variance Decomposition (FEVD): % of forecast variance explained

Model:
  For 2 variables y1, y2:
  y1_t = c1 + φ11*y1_{t-1} + φ12*y2_{t-1} + ... + ε1_t
  y2_t = c2 + φ21*y1_{t-1} + φ22*y2_{t-1} + ... + ε2_t

Use Cases:
  - Forecasting multiple assets simultaneously
  - Understanding cross-market relationships
  - Analyzing how shocks propagate (e.g., VIX shock → SP500 impact)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

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
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "var"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]

# Common asset combinations for VAR
VAR_COMBINATIONS = [
    ["SP500", "NASDAQ", "VIX"],      # Major US indices + volatility
    ["SP500", "GOLD", "VIX"],        # Equity, safe haven, volatility
    ["BTC", "GOLD"],                 # Alternative stores of value
    ["SP500", "TESLA"],              # Market and high-beta stock
]


# ── Evaluation Metrics ─────────────────────────────────────────────────────────
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate RMSE, MAE, MAPE."""
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    if len(actual) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan}

    errors = actual - predicted
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))

    nonzero = actual != 0
    mape = np.mean(np.abs(errors[nonzero]) / np.abs(actual[nonzero])) * 100 if nonzero.sum() > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape}


# ── Data Preparation ───────────────────────────────────────────────────────────
def load_multi_asset_data(
    assets: list[str],
    column: str = "Close"
) -> pd.DataFrame:
    """
    Load and align data for multiple assets.

    Parameters
    ----------
    assets : list[str]
        List of asset names to load
    column : str
        Column to use (default: 'Close')

    Returns
    -------
    pd.DataFrame with assets as columns, aligned by date
    """
    dfs = {}

    for name in assets:
        path = PROCESSED_DIR / f"{name}_features.parquet"
        if not path.exists():
            log.warning(f"[{name}] Not found — skipping")
            continue

        df = pd.read_parquet(path)
        dfs[name] = df[column]

    # Combine and align dates
    combined = pd.DataFrame(dfs)
    combined = combined.dropna()

    log.info(f"Loaded {len(assets)} assets: {len(combined)} aligned observations")
    return combined


def ensure_stationarity(
    data: pd.DataFrame,
    max_d: int = 2
) -> tuple[pd.DataFrame, dict]:
    """
    Make all series stationary via differencing.

    VAR requires stationary series for valid inference.

    Returns
    -------
    tuple of (differenced DataFrame, dict of d values per asset)
    """
    d_values = {}
    stationary_data = pd.DataFrame(index=data.index)

    for col in data.columns:
        series = data[col].copy()
        d = 0

        # Test and difference until stationary
        for d in range(max_d + 1):
            if d > 0:
                series = series.diff()

            result = adfuller(series.dropna())
            if result[1] < 0.05:  # p-value < 0.05 = stationary
                break

        d_values[col] = d
        stationary_data[col] = series

    stationary_data = stationary_data.dropna()
    log.info(f"Stationarity: differencing applied = {d_values}")

    return stationary_data, d_values


# ── VAR Model Fitting ──────────────────────────────────────────────────────────
def fit_var(
    data: pd.DataFrame,
    max_lags: int = 15,
    ic: str = "aic"
) -> dict:
    """
    Fit VAR model with automatic lag selection.

    Parameters
    ----------
    data : pd.DataFrame
        Multivariate stationary data
    max_lags : int
        Maximum lags to consider
    ic : str
        Information criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')

    Returns
    -------
    dict with fitted model and diagnostics
    """
    model = VAR(data)

    # Select optimal lag order
    lag_order = model.select_order(maxlags=max_lags)
    optimal_lag = getattr(lag_order, ic)

    log.info(f"VAR lag selection:")
    log.info(f"  AIC: {lag_order.aic}, BIC: {lag_order.bic}, HQIC: {lag_order.hqic}")
    log.info(f"  Selected: p={optimal_lag} (by {ic})")

    # Fit model
    fitted = model.fit(optimal_lag)

    return {
        "model": fitted,
        "lag_order": optimal_lag,
        "lag_selection": lag_order,
        "aic": fitted.aic,
        "bic": fitted.bic,
        "fpe": fitted.fpe,
        "variables": list(data.columns),
        "n_obs": len(data),
    }


def var_forecast(
    fitted_result: dict,
    horizon: int,
    data: pd.DataFrame
) -> dict:
    """
    Generate VAR forecasts.

    Parameters
    ----------
    fitted_result : dict
        Result from fit_var()
    horizon : int
        Forecast horizon
    data : pd.DataFrame
        Original data (for getting last values)

    Returns
    -------
    dict with forecasts and confidence intervals
    """
    model = fitted_result["model"]
    lag_order = fitted_result["lag_order"]

    # Get last lag_order observations for forecasting
    y_input = data.values[-lag_order:]

    # Forecast
    forecast = model.forecast(y_input, steps=horizon)

    # Forecast with confidence intervals
    forecast_ci = model.forecast_interval(y_input, steps=horizon, alpha=0.05)

    # Create DataFrames
    last_date = data.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq="D")[1:]

    forecast_df = pd.DataFrame(
        forecast,
        index=forecast_index,
        columns=data.columns
    )

    lower_df = pd.DataFrame(
        forecast_ci[1],
        index=forecast_index,
        columns=data.columns
    )

    upper_df = pd.DataFrame(
        forecast_ci[2],
        index=forecast_index,
        columns=data.columns
    )

    log.info(f"VAR forecast generated: {horizon} steps for {list(data.columns)}")

    return {
        "forecast": forecast_df,
        "lower": lower_df,
        "upper": upper_df,
        "horizon": horizon,
    }


# ── Granger Causality ──────────────────────────────────────────────────────────
def test_granger_causality(
    data: pd.DataFrame,
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Test Granger causality between all pairs of variables.

    H0: X does not Granger-cause Y
    If p < 0.05, X Granger-causes Y (past X helps predict Y)

    Parameters
    ----------
    data : pd.DataFrame
        Stationary multivariate data
    max_lag : int
        Maximum lag to test

    Returns
    -------
    pd.DataFrame with causality results (Causing → Caused, p-value)
    """
    variables = data.columns
    results = []

    for caused in variables:
        for causing in variables:
            if caused == causing:
                continue

            # Test if 'causing' Granger-causes 'caused'
            test_data = data[[caused, causing]].dropna()

            try:
                gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                # Get min p-value across all lags (most significant)
                min_pvalue = min(gc_result[lag][0]["ssr_chi2test"][1] for lag in range(1, max_lag + 1))
                best_lag = min(range(1, max_lag + 1),
                              key=lambda lag: gc_result[lag][0]["ssr_chi2test"][1])

                results.append({
                    "Causing": causing,
                    "Caused": caused,
                    "p_value": min_pvalue,
                    "Best_Lag": best_lag,
                    "Granger_Causes": min_pvalue < 0.05,
                })

            except Exception as e:
                log.warning(f"Granger test failed: {causing} → {caused}: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("p_value")

    # Log significant relationships
    significant = results_df[results_df["Granger_Causes"]]
    log.info(f"Granger Causality - {len(significant)} significant relationships found:")
    for _, row in significant.iterrows():
        log.info(f"  {row['Causing']} → {row['Caused']} (p={row['p_value']:.4f}, lag={row['Best_Lag']})")

    return results_df


# ── Impulse Response Function ──────────────────────────────────────────────────
def compute_irf(
    fitted_result: dict,
    periods: int = 20
) -> dict:
    """
    Compute Impulse Response Functions.

    Shows how a shock in one variable affects all variables over time.

    Parameters
    ----------
    fitted_result : dict
        Result from fit_var()
    periods : int
        Number of periods to compute IRF

    Returns
    -------
    dict with IRF results
    """
    model = fitted_result["model"]
    irf = model.irf(periods)

    return {
        "irf": irf,
        "periods": periods,
        "variables": fitted_result["variables"],
    }


def plot_irf(
    irf_result: dict,
    name: str = "VAR",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot Impulse Response Functions."""
    irf = irf_result["irf"]

    fig = irf.plot(orth=True, figsize=(14, 10))
    fig.suptitle(f"{name} - Impulse Response Functions", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved IRF plot → {save_path.name}")

    return fig


# ── Forecast Error Variance Decomposition ──────────────────────────────────────
def compute_fevd(
    fitted_result: dict,
    periods: int = 20
) -> dict:
    """
    Compute Forecast Error Variance Decomposition.

    Shows what percentage of forecast error variance is explained by each variable.

    Parameters
    ----------
    fitted_result : dict
        Result from fit_var()
    periods : int
        Number of periods to decompose

    Returns
    -------
    dict with FEVD results
    """
    model = fitted_result["model"]
    fevd = model.fevd(periods)

    return {
        "fevd": fevd,
        "periods": periods,
        "variables": fitted_result["variables"],
    }


def plot_fevd(
    fevd_result: dict,
    name: str = "VAR",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot Forecast Error Variance Decomposition."""
    fevd = fevd_result["fevd"]

    fig = fevd.plot(figsize=(14, 10))
    fig.suptitle(f"{name} - Forecast Error Variance Decomposition", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved FEVD plot → {save_path.name}")

    return fig


# ── Visualization ──────────────────────────────────────────────────────────────
def plot_var_forecast(
    data: pd.DataFrame,
    test: pd.DataFrame,
    forecast_result: dict,
    name: str,
    show_last_n: int = 60,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot VAR forecasts for all variables."""
    n_vars = len(data.columns)
    fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars))
    fig.suptitle(f"{name} - VAR Multi-Asset Forecast", fontsize=14, fontweight="bold")

    if n_vars == 1:
        axes = [axes]

    forecast_df = forecast_result["forecast"]
    lower_df = forecast_result["lower"]
    upper_df = forecast_result["upper"]

    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))

    for idx, col in enumerate(data.columns):
        ax = axes[idx]
        color = colors[idx]

        # Training data
        train_plot = data[col].iloc[-show_last_n:]
        ax.plot(train_plot.index, train_plot.values, label="Training", color="gray", alpha=0.7)

        # Test data (if available)
        if col in test.columns:
            ax.plot(test.index, test[col].values, label="Actual", color="black", linewidth=2)

        # Forecast
        ax.plot(forecast_df.index, forecast_df[col].values,
                label="Forecast", color=color, linewidth=2, linestyle="--")

        # Confidence interval
        ax.fill_between(forecast_df.index,
                        lower_df[col].values,
                        upper_df[col].values,
                        alpha=0.2, color=color, label="95% CI")

        ax.set_title(f"{col}")
        ax.legend(loc="upper left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved VAR forecast plot → {save_path.name}")

    return fig


# ── Main Analysis Function ─────────────────────────────────────────────────────
def analyze_var(
    assets: list[str],
    test_size: int = 30,
    max_lags: int = 15,
    save_plots: bool = True
) -> dict:
    """
    Complete VAR analysis for a set of assets.

    Parameters
    ----------
    assets : list[str]
        List of asset names to include
    test_size : int
        Number of observations for testing
    max_lags : int
        Maximum lags for model selection
    save_plots : bool
        Whether to save plots

    Returns
    -------
    dict with all VAR analysis results
    """
    name = "_".join(assets)
    log.info(f"\n{'='*60}")
    log.info(f"VAR Analysis: {assets}")
    log.info(f"{'='*60}")

    # Load and align data
    data = load_multi_asset_data(assets)
    if data.empty or len(data.columns) < 2:
        log.error("Need at least 2 assets for VAR")
        return {"success": False, "error": "Insufficient data"}

    # Train/test split
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    horizon = len(test)

    log.info(f"Train: {len(train)} obs, Test: {len(test)} obs")

    # Make stationary
    stationary_train, d_values = ensure_stationarity(train)

    # Fit VAR
    var_result = fit_var(stationary_train, max_lags=max_lags)

    # Forecast
    forecast_result = var_forecast(var_result, horizon, stationary_train)

    # Granger causality
    granger_result = test_granger_causality(stationary_train, max_lag=min(10, var_result["lag_order"]))

    # IRF
    irf_result = compute_irf(var_result, periods=20)

    # FEVD
    fevd_result = compute_fevd(var_result, periods=20)

    # Evaluate forecasts (on differenced data)
    metrics = {}
    stationary_test, _ = ensure_stationarity(test)
    for col in stationary_test.columns:
        if col in forecast_result["forecast"].columns:
            actual = stationary_test[col].values[:len(forecast_result["forecast"])]
            predicted = forecast_result["forecast"][col].values
            metrics[col] = calculate_metrics(actual, predicted)

    # Summary
    summary = f"""
    VAR Analysis: {assets}
    ─────────────────────────────
    Lag Order: {var_result['lag_order']} (selected by AIC)
    Variables: {var_result['variables']}
    Observations: {var_result['n_obs']}
    AIC: {var_result['aic']:.2f}

    Granger Causality (significant):
    {granger_result[granger_result['Granger_Causes']][['Causing', 'Caused', 'p_value']].to_string(index=False) if len(granger_result[granger_result['Granger_Causes']]) > 0 else '  None found'}

    Forecast Metrics (differenced):
    {pd.DataFrame(metrics).T.to_string()}
    """
    log.info(summary)

    # Save plots
    if save_plots:
        # Forecast plot
        forecast_path = RESULTS_DIR / f"{name}_var_forecast.png"
        plot_var_forecast(train, test, forecast_result, name, save_path=forecast_path)
        plt.close()

        # IRF plot
        irf_path = RESULTS_DIR / f"{name}_var_irf.png"
        plot_irf(irf_result, name, save_path=irf_path)
        plt.close()

        # FEVD plot
        fevd_path = RESULTS_DIR / f"{name}_var_fevd.png"
        plot_fevd(fevd_result, name, save_path=fevd_path)
        plt.close()

    return {
        "assets": assets,
        "var_result": var_result,
        "forecast_result": forecast_result,
        "granger_causality": granger_result,
        "irf_result": irf_result,
        "fevd_result": fevd_result,
        "metrics": metrics,
        "d_values": d_values,
        "train": train,
        "test": test,
        "summary": summary,
        "success": True,
    }


def analyze_all_combinations(test_size: int = 30) -> dict:
    """
    Run VAR analysis on all predefined asset combinations.
    """
    log.info("=" * 60)
    log.info("VAR Analysis - All Asset Combinations")
    log.info("=" * 60)

    results = {}
    summary_rows = []

    for assets in VAR_COMBINATIONS:
        name = "_".join(assets)
        result = analyze_var(assets, test_size=test_size)
        results[name] = result

        if result.get("success"):
            summary_rows.append({
                "Combination": name,
                "Variables": len(assets),
                "Lag_Order": result["var_result"]["lag_order"],
                "AIC": result["var_result"]["aic"],
                "Granger_Links": len(result["granger_causality"][result["granger_causality"]["Granger_Causes"]]),
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "var_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path.name}")

    print("\n" + "=" * 60)
    print("VAR ANALYSIS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = analyze_all_combinations(test_size=30)
