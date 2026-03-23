"""
exponential_smoothing.py
========================
Exponential smoothing methods for time series forecasting.

Implements 4 methods from the exponential smoothing family:

  1. Simple Exponential Smoothing (SES)
     - For data with no trend and no seasonality
     - Single smoothing parameter α (alpha)
     - Level equation: ℓ_t = α × y_t + (1-α) × ℓ_{t-1}

  2. Holt's Linear Trend Method
     - For data with trend but no seasonality
     - Two parameters: α (level) and β (trend)
     - Level equation: ℓ_t = α × y_t + (1-α) × (ℓ_{t-1} + b_{t-1})
     - Trend equation: b_t = β × (ℓ_t - ℓ_{t-1}) + (1-β) × b_{t-1}

  3. Damped Trend Method
     - Holt's method with damping parameter φ (phi)
     - Prevents over-forecasting with long horizons
     - Trend decays toward zero over time

  4. Holt-Winters Seasonal Method
     - For data with both trend and seasonality
     - Three parameters: α, β, γ (seasonal)
     - Two variants: additive and multiplicative

ETS Framework:
  Exponential smoothing methods are special cases of ETS models:
  - E: Error type (Additive or Multiplicative)
  - T: Trend type (None, Additive, Additive damped, Multiplicative)
  - S: Seasonal type (None, Additive, Multiplicative)

  Examples:
  - SES = ETS(A,N,N) - Additive errors, No trend, No seasonality
  - Holt = ETS(A,A,N) - Additive errors, Additive trend, No seasonality
  - Holt-Winters = ETS(A,A,A) or ETS(A,A,M)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

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
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "exponential_smoothing"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]


# ── Evaluation Metrics ─────────────────────────────────────────────────────────
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate RMSE, MAE, MAPE, SMAPE."""
    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[mask], predicted[mask]

    if len(actual) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "smape": np.nan}

    errors = actual - predicted
    abs_errors = np.abs(errors)

    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)

    nonzero_mask = actual != 0
    mape = np.mean(abs_errors[nonzero_mask] / np.abs(actual[nonzero_mask])) * 100 if nonzero_mask.sum() > 0 else np.nan

    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    nonzero_denom = denominator != 0
    smape = np.mean(abs_errors[nonzero_denom] / denominator[nonzero_denom]) * 100 if nonzero_denom.sum() > 0 else np.nan

    return {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape}


# ── Method 1: Simple Exponential Smoothing ─────────────────────────────────────
def ses_forecast(
    train: pd.Series,
    horizon: int,
    alpha: Optional[float] = None,
    confidence: float = 0.95
) -> dict:
    """
    Simple Exponential Smoothing (SES).

    Best for: series with no trend or seasonality.
    Equivalent to: ARIMA(0,1,1)

    Parameters
    ----------
    train : pd.Series
        Training data
    horizon : int
        Forecast horizon
    alpha : float, optional
        Smoothing parameter (0-1). If None, optimized by AIC.
    confidence : float
        Confidence level for prediction intervals

    Returns
    -------
    dict with forecast, fitted model, and parameters
    """
    # Fit model
    if alpha is not None:
        model = SimpleExpSmoothing(train, initialization_method="estimated")
        fitted = model.fit(smoothing_level=alpha, optimized=False)
    else:
        model = SimpleExpSmoothing(train, initialization_method="estimated")
        fitted = model.fit(optimized=True)

    # Forecast
    forecast = fitted.forecast(horizon)

    # Prediction intervals (simulation-based)
    # For SES, forecast variance grows: σ² × [1 + (h-1) × α²]
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    residuals = train - fitted.fittedvalues
    sigma = residuals.std()
    alpha_opt = fitted.params.get("smoothing_level", 0.5)

    h = np.arange(1, horizon + 1)
    se = sigma * np.sqrt(1 + (h - 1) * alpha_opt ** 2)

    lower = forecast - z * se
    upper = forecast + z * se

    log.info(f"SES: alpha={alpha_opt:.4f}, AIC={fitted.aic:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fitted_model": fitted,
        "fittedvalues": fitted.fittedvalues,
        "method": "SES",
        "params": {"alpha": alpha_opt},
        "aic": fitted.aic,
        "bic": fitted.bic,
    }


# ── Method 2: Holt's Linear Trend ──────────────────────────────────────────────
def holt_forecast(
    train: pd.Series,
    horizon: int,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    confidence: float = 0.95
) -> dict:
    """
    Holt's Linear Trend Method.

    Best for: series with trend but no seasonality.

    Parameters
    ----------
    train : pd.Series
        Training data
    horizon : int
        Forecast horizon
    alpha : float, optional
        Level smoothing (0-1)
    beta : float, optional
        Trend smoothing (0-1)
    confidence : float
        Confidence level

    Returns
    -------
    dict with forecast and model details
    """
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal=None,
        initialization_method="estimated"
    )

    if alpha is not None and beta is not None:
        fitted = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    else:
        fitted = model.fit(optimized=True)

    forecast = fitted.forecast(horizon)

    # Prediction intervals
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    residuals = train - fitted.fittedvalues
    sigma = residuals.std()

    h = np.arange(1, horizon + 1)
    # Approximate SE for Holt's method
    se = sigma * np.sqrt(1 + h * 0.1)  # Simplified approximation

    lower = forecast - z * se
    upper = forecast + z * se

    alpha_opt = fitted.params.get("smoothing_level", 0.5)
    beta_opt = fitted.params.get("smoothing_trend", 0.1)

    log.info(f"Holt: alpha={alpha_opt:.4f}, beta={beta_opt:.4f}, AIC={fitted.aic:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fitted_model": fitted,
        "fittedvalues": fitted.fittedvalues,
        "method": "Holt",
        "params": {"alpha": alpha_opt, "beta": beta_opt},
        "aic": fitted.aic,
        "bic": fitted.bic,
    }


# ── Method 3: Damped Trend ─────────────────────────────────────────────────────
def damped_trend_forecast(
    train: pd.Series,
    horizon: int,
    phi: Optional[float] = None,
    confidence: float = 0.95
) -> dict:
    """
    Damped Trend Method (Holt with damping).

    Best for: series with trend that is expected to level off.
    Damping parameter φ controls how quickly trend decays.

    Parameters
    ----------
    train : pd.Series
        Training data
    horizon : int
        Forecast horizon
    phi : float, optional
        Damping parameter (0.8-1.0 typical). If None, optimized.
    confidence : float
        Confidence level

    Returns
    -------
    dict with forecast and model details
    """
    model = ExponentialSmoothing(
        train,
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    )

    if phi is not None:
        fitted = model.fit(damping_trend=phi, optimized=True)
    else:
        fitted = model.fit(optimized=True)

    forecast = fitted.forecast(horizon)

    # Prediction intervals
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    residuals = train - fitted.fittedvalues
    sigma = residuals.std()

    h = np.arange(1, horizon + 1)
    se = sigma * np.sqrt(1 + h * 0.05)  # Lower growth due to damping

    lower = forecast - z * se
    upper = forecast + z * se

    alpha_opt = fitted.params.get("smoothing_level", 0.5)
    beta_opt = fitted.params.get("smoothing_trend", 0.1)
    phi_opt = fitted.params.get("damping_trend", 0.98)

    log.info(f"Damped: alpha={alpha_opt:.4f}, beta={beta_opt:.4f}, phi={phi_opt:.4f}, AIC={fitted.aic:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fitted_model": fitted,
        "fittedvalues": fitted.fittedvalues,
        "method": "Damped",
        "params": {"alpha": alpha_opt, "beta": beta_opt, "phi": phi_opt},
        "aic": fitted.aic,
        "bic": fitted.bic,
    }


# ── Method 4: Holt-Winters Seasonal ────────────────────────────────────────────
def holt_winters_forecast(
    train: pd.Series,
    horizon: int,
    seasonal_period: int = 21,
    seasonal: Literal["add", "mul"] = "add",
    confidence: float = 0.95
) -> dict:
    """
    Holt-Winters Seasonal Method.

    Best for: series with both trend and seasonality.

    Parameters
    ----------
    train : pd.Series
        Training data (needs at least 2 full seasonal cycles)
    horizon : int
        Forecast horizon
    seasonal_period : int
        Number of periods in one season (21 = monthly for trading days)
    seasonal : str
        'add' for additive, 'mul' for multiplicative seasonality
    confidence : float
        Confidence level

    Returns
    -------
    dict with forecast and model details
    """
    # Need at least 2 seasonal cycles
    if len(train) < 2 * seasonal_period:
        log.warning(f"Holt-Winters: Not enough data for seasonal_period={seasonal_period}")
        seasonal_period = max(7, len(train) // 4)

    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal=seasonal,
            seasonal_periods=seasonal_period,
            initialization_method="estimated"
        )
        fitted = model.fit(optimized=True)

    except Exception as e:
        log.warning(f"Holt-Winters failed: {e}. Falling back to Holt.")
        return holt_forecast(train, horizon, confidence=confidence)

    forecast = fitted.forecast(horizon)

    # Prediction intervals
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)

    residuals = train - fitted.fittedvalues
    sigma = residuals.std()

    h = np.arange(1, horizon + 1)
    se = sigma * np.sqrt(1 + h * 0.08)

    lower = forecast - z * se
    upper = forecast + z * se

    alpha_opt = fitted.params.get("smoothing_level", 0.5)
    beta_opt = fitted.params.get("smoothing_trend", 0.1)
    gamma_opt = fitted.params.get("smoothing_seasonal", 0.1)

    log.info(f"Holt-Winters ({seasonal}): alpha={alpha_opt:.4f}, beta={beta_opt:.4f}, "
             f"gamma={gamma_opt:.4f}, m={seasonal_period}, AIC={fitted.aic:.2f}")

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fitted_model": fitted,
        "fittedvalues": fitted.fittedvalues,
        "method": f"Holt-Winters ({seasonal})",
        "params": {
            "alpha": alpha_opt,
            "beta": beta_opt,
            "gamma": gamma_opt,
            "seasonal_period": seasonal_period,
        },
        "aic": fitted.aic,
        "bic": fitted.bic,
    }


# ── Run All Exponential Smoothing Methods ──────────────────────────────────────
def run_all_exp_smoothing(
    train: pd.Series,
    test: pd.Series,
    seasonal_period: int = 21,
    name: str = "Series",
    horizon: Optional[int] = None
) -> dict:
    """
    Run all exponential smoothing methods and compare.
    """
    horizon = horizon if horizon is not None else len(test)
    log.info(f"\n[{name}] Running exponential smoothing methods — horizon={horizon}")

    methods = {}

    # SES
    try:
        methods["SES"] = ses_forecast(train, horizon)
    except Exception as e:
        log.warning(f"SES failed: {e}")

    # Holt
    try:
        methods["Holt"] = holt_forecast(train, horizon)
    except Exception as e:
        log.warning(f"Holt failed: {e}")

    # Damped
    try:
        methods["Damped"] = damped_trend_forecast(train, horizon)
    except Exception as e:
        log.warning(f"Damped failed: {e}")

    # Holt-Winters (additive)
    try:
        methods["Holt-Winters"] = holt_winters_forecast(train, horizon, seasonal_period)
    except Exception as e:
        log.warning(f"Holt-Winters failed: {e}")

    comparison = []
    for method_name, result in methods.items():
        if len(test) > 0:
            metrics = calculate_metrics(test.values, result["forecast"].values)
            result["metrics"] = metrics
        else:
            metrics = {"rmse": np.nan, "mae": np.nan, "mape": np.nan}
            
        comparison.append({
            "Method": method_name,
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "MAPE": metrics["mape"],
            "AIC": result.get("aic", np.nan),
            "BIC": result.get("bic", np.nan),
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values("RMSE")
    comparison_df["Rank"] = range(1, len(comparison_df) + 1)

    best_method = comparison_df.iloc[0]["Method"] if len(comparison_df) > 0 else None
    log.info(f"[{name}] Best exp. smoothing: {best_method}")

    return {
        "methods": methods,
        "comparison": comparison_df,
        "best_method": best_method,
        "test_series": test,
        "train_series": train,
    }


# ── Visualization ──────────────────────────────────────────────────────────────
def plot_exp_smoothing_forecasts(
    results: dict,
    name: str,
    show_last_n: int = 100,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot all exponential smoothing forecasts."""
    train = results["train_series"]
    test = results["test_series"]
    methods = results["methods"]

    n_methods = len(methods)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} - Exponential Smoothing Methods", fontsize=14, fontweight="bold")

    axes_flat = axes.flatten()
    colors = ["blue", "green", "orange", "red"]

    for idx, (method_name, result) in enumerate(methods.items()):
        if idx >= 4:
            break
        ax = axes_flat[idx]
        color = colors[idx % len(colors)]

        # Training data
        train_plot = train.iloc[-show_last_n:]
        ax.plot(train_plot.index, train_plot.values, label="Training", color="gray", alpha=0.7)

        # Actual test
        ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)

        # Forecast
        ax.plot(result["forecast"].index, result["forecast"].values,
                label="Forecast", color=color, linewidth=2, linestyle="--")

        # Confidence interval
        ax.fill_between(result["forecast"].index,
                        result["lower"].values,
                        result["upper"].values,
                        alpha=0.2, color=color, label="95% CI")

        metrics = result.get("metrics", {})
        ax.set_title(f"{method_name}\n"
                    f"RMSE={metrics.get('rmse', 0):.2f}, AIC={result.get('aic', 0):.0f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

    # Hide unused subplots
    for idx in range(n_methods, 4):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved plot → {save_path.name}")

    return fig


def plot_fitted_vs_actual(
    results: dict,
    name: str,
    method_name: str = "Holt",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Plot fitted values vs actual for a specific method."""
    if method_name not in results["methods"]:
        log.warning(f"Method {method_name} not found")
        return None

    train = results["train_series"]
    result = results["methods"][method_name]
    fitted = result["fittedvalues"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"{name} - {method_name} Fitted vs Actual", fontsize=14, fontweight="bold")

    # Top: Fitted vs Actual
    ax = axes[0]
    ax.plot(train.index, train.values, label="Actual", color="black", linewidth=1)
    ax.plot(fitted.index, fitted.values, label="Fitted", color="blue", alpha=0.8)
    ax.legend()
    ax.set_title("Fitted vs Actual")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    # Bottom: Residuals
    ax = axes[1]
    residuals = train - fitted
    ax.plot(residuals.index, residuals.values, color="red", alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--")
    ax.fill_between(residuals.index, residuals.values, 0, alpha=0.3, color="red")
    ax.set_title(f"Residuals (std={residuals.std():.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ── Main Analysis Function ─────────────────────────────────────────────────────
def analyze_exp_smoothing(
    series: pd.Series,
    name: str,
    test_size: int = 30,
    seasonal_period: int = 21,
    save_plots: bool = True
) -> dict:
    """Complete exponential smoothing analysis for an asset."""
    log.info(f"\n{'='*60}")
    log.info(f"[{name}] Exponential Smoothing Analysis")
    log.info(f"{'='*60}")

    # Train/test split
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    log.info(f"[{name}] Train: {len(train)} obs, Test: {len(test)} obs")

    # Run all methods
    results = run_all_exp_smoothing(train, test, seasonal_period, name)

    # Summary
    summary = f"""
    Exponential Smoothing Analysis: {name}
    ─────────────────────────────────────
    Train: {len(train)} observations
    Test:  {len(test)} observations

    Method Comparison (by RMSE):
{results['comparison'].to_string(index=False)}

    Best Method: {results['best_method']}
    """
    log.info(summary)

    # Save plots
    if save_plots:
        plot_path = RESULTS_DIR / f"{name}_exp_smoothing.png"
        plot_exp_smoothing_forecasts(results, name, save_path=plot_path)
        plt.close()

        if results["best_method"]:
            fitted_path = RESULTS_DIR / f"{name}_fitted_{results['best_method']}.png"
            plot_fitted_vs_actual(results, name, results["best_method"], fitted_path)
            plt.close()

    results["summary"] = summary
    return results


def analyze_all_assets(test_size: int = 30) -> dict:
    """Run exponential smoothing analysis on all assets."""
    log.info("=" * 60)
    log.info("Exponential Smoothing Analysis - All Assets")
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

        result = analyze_exp_smoothing(series, name, test_size=test_size)
        results[name] = result

        for _, row in result["comparison"].iterrows():
            summary_rows.append({
                "Asset": name,
                "Method": row["Method"],
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "AIC": row["AIC"],
                "Rank": row["Rank"],
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "exp_smoothing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path.name}")

    # Best method by asset
    best_by_asset = summary_df[summary_df["Rank"] == 1][["Asset", "Method", "RMSE", "AIC"]]
    print("\n" + "=" * 60)
    print("EXPONENTIAL SMOOTHING - BEST METHOD BY ASSET")
    print("=" * 60)
    print(best_by_asset.to_string(index=False))
    print("=" * 60)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = analyze_all_assets(test_size=30)
