"""
arima_models.py
===============
ARIMA and SARIMA models for time series forecasting.

Implements:
  - ARIMA(p,d,q) model fitting and forecasting
  - SARIMA(p,d,q)(P,D,Q,m) for seasonal time series
  - Grid search for optimal order selection
  - Model diagnostics (residual analysis, Ljung-Box test)
  - Forecasting with prediction intervals

ARIMA Components:
  - AR(p): Autoregressive - uses p past values
  - I(d): Integrated - d times differenced
  - MA(q): Moving Average - uses q past forecast errors

Model Selection:
  - Use ACF/PACF plots to determine p and q
  - Use stationarity tests to determine d
  - Use AIC/BIC for final model selection

SARIMA Extension:
  - (P,D,Q)_m: Seasonal component with period m
  - P: Seasonal AR order
  - D: Seasonal differencing
  - Q: Seasonal MA order
  - m: Seasonal period (21 for monthly trading days, 252 for yearly)
"""

import logging
import warnings
from pathlib import Path
from typing import Optional
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "arima"
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


# ── ARIMA Model ────────────────────────────────────────────────────────────────
def fit_arima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    name: str = "Series"
) -> dict:
    """
    Fit ARIMA(p,d,q) model.

    Parameters
    ----------
    train : pd.Series
        Training data
    order : tuple
        (p, d, q) - AR order, differencing order, MA order
    name : str
        Name for logging

    Returns
    -------
    dict with fitted model, parameters, and diagnostics
    """
    p, d, q = order

    try:
        model = ARIMA(train, order=order)
        fitted = model.fit()

        log.info(f"[{name}] ARIMA{order}: AIC={fitted.aic:.2f}, BIC={fitted.bic:.2f}")

        return {
            "model": fitted,
            "order": order,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "params": fitted.params,
            "resid": fitted.resid,
            "fittedvalues": fitted.fittedvalues,
            "success": True,
        }

    except Exception as e:
        log.warning(f"[{name}] ARIMA{order} failed: {e}")
        return {
            "model": None,
            "order": order,
            "aic": np.inf,
            "bic": np.inf,
            "success": False,
            "error": str(e),
        }


def arima_forecast(
    train: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
    confidence: float = 0.95,
    name: str = "Series"
) -> dict:
    """
    Fit ARIMA and generate forecast with prediction intervals.
    """
    fit_result = fit_arima(train, order, name)

    if not fit_result["success"]:
        return {
            "forecast": pd.Series([np.nan] * horizon),
            "lower": pd.Series([np.nan] * horizon),
            "upper": pd.Series([np.nan] * horizon),
            "success": False,
            **fit_result,
        }

    fitted = fit_result["model"]

    # Generate forecast
    forecast_result = fitted.get_forecast(steps=horizon)
    forecast = forecast_result.predicted_mean

    # Prediction intervals
    conf_int = forecast_result.conf_int(alpha=1 - confidence)
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fittedvalues": fit_result["fittedvalues"],
        "resid": fit_result["resid"],
        "method": f"ARIMA{order}",
        "success": True,
        **fit_result,
    }


# ── SARIMA Model ───────────────────────────────────────────────────────────────
def fit_sarima(
    train: pd.Series,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 21),
    name: str = "Series"
) -> dict:
    """
    Fit SARIMA(p,d,q)(P,D,Q,m) model.

    Parameters
    ----------
    train : pd.Series
        Training data
    order : tuple
        (p, d, q) - non-seasonal orders
    seasonal_order : tuple
        (P, D, Q, m) - seasonal orders and period
    name : str
        Name for logging

    Returns
    -------
    dict with fitted model and diagnostics
    """
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        fitted = model.fit(disp=False)

        log.info(f"[{name}] SARIMA{order}x{seasonal_order}: AIC={fitted.aic:.2f}")

        return {
            "model": fitted,
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "params": fitted.params,
            "resid": fitted.resid,
            "fittedvalues": fitted.fittedvalues,
            "success": True,
        }

    except Exception as e:
        log.warning(f"[{name}] SARIMA{order}x{seasonal_order} failed: {e}")
        return {
            "model": None,
            "order": order,
            "seasonal_order": seasonal_order,
            "aic": np.inf,
            "bic": np.inf,
            "success": False,
            "error": str(e),
        }


def sarima_forecast(
    train: pd.Series,
    horizon: int,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 21),
    confidence: float = 0.95,
    name: str = "Series"
) -> dict:
    """
    Fit SARIMA and generate forecast with prediction intervals.
    """
    fit_result = fit_sarima(train, order, seasonal_order, name)

    if not fit_result["success"]:
        return {
            "forecast": pd.Series([np.nan] * horizon),
            "lower": pd.Series([np.nan] * horizon),
            "upper": pd.Series([np.nan] * horizon),
            "success": False,
            **fit_result,
        }

    fitted = fit_result["model"]

    forecast_result = fitted.get_forecast(steps=horizon)
    forecast = forecast_result.predicted_mean

    conf_int = forecast_result.conf_int(alpha=1 - confidence)
    lower = conf_int.iloc[:, 0]
    upper = conf_int.iloc[:, 1]

    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "fittedvalues": fit_result["fittedvalues"],
        "resid": fit_result["resid"],
        "method": f"SARIMA{order}x{seasonal_order}",
        "success": True,
        **fit_result,
    }


# ── Grid Search for Optimal Orders ─────────────────────────────────────────────
def grid_search_arima(
    train: pd.Series,
    p_range: range = range(0, 4),
    d_range: range = range(0, 3),
    q_range: range = range(0, 4),
    criterion: str = "aic",
    name: str = "Series"
) -> dict:
    """
    Grid search over ARIMA orders to find optimal model.

    Parameters
    ----------
    train : pd.Series
        Training data
    p_range, d_range, q_range : range
        Ranges for AR, differencing, and MA orders
    criterion : str
        'aic' or 'bic' for model selection
    name : str
        Name for logging

    Returns
    -------
    dict with best model and all tried combinations
    """
    log.info(f"[{name}] Grid search: p={list(p_range)}, d={list(d_range)}, q={list(q_range)}")

    results = []
    total = len(p_range) * len(d_range) * len(q_range)

    for idx, (p, d, q) in enumerate(product(p_range, d_range, q_range)):
        order = (p, d, q)

        # Skip degenerate models
        if p == 0 and q == 0:
            continue

        fit_result = fit_arima(train, order, name)

        results.append({
            "order": order,
            "p": p,
            "d": d,
            "q": q,
            "aic": fit_result["aic"],
            "bic": fit_result["bic"],
            "success": fit_result["success"],
        })

        if (idx + 1) % 10 == 0:
            log.info(f"[{name}] Progress: {idx + 1}/{total} models tested")

    # Convert to DataFrame and find best
    results_df = pd.DataFrame(results)
    successful = results_df[results_df["success"]]

    if len(successful) == 0:
        log.warning(f"[{name}] No successful models found!")
        return {
            "best_order": None,
            "best_aic": np.inf,
            "best_bic": np.inf,
            "all_results": results_df,
            "top_5": pd.DataFrame(),
        }

    # Sort by criterion
    successful = successful.sort_values(criterion)
    best_row = successful.iloc[0]
    best_order = best_row["order"]

    log.info(f"[{name}] Best order: ARIMA{best_order} ({criterion}={best_row[criterion]:.2f})")

    return {
        "best_order": best_order,
        "best_aic": best_row["aic"],
        "best_bic": best_row["bic"],
        "all_results": results_df,
        "top_5": successful.head(5),
    }


def grid_search_sarima(
    train: pd.Series,
    p_range: range = range(0, 3),
    d_range: range = range(0, 2),
    q_range: range = range(0, 3),
    P_range: range = range(0, 2),
    D_range: range = range(0, 2),
    Q_range: range = range(0, 2),
    m: int = 21,
    criterion: str = "aic",
    name: str = "Series"
) -> dict:
    """
    Grid search over SARIMA orders.
    """
    log.info(f"[{name}] SARIMA grid search with m={m}")

    results = []
    orders_to_try = list(product(p_range, d_range, q_range, P_range, D_range, Q_range))
    total = len(orders_to_try)

    for idx, (p, d, q, P, D, Q) in enumerate(orders_to_try):
        order = (p, d, q)
        seasonal_order = (P, D, Q, m)

        # Skip degenerate models
        if p == 0 and q == 0 and P == 0 and Q == 0:
            continue

        fit_result = fit_sarima(train, order, seasonal_order, name)

        results.append({
            "order": order,
            "seasonal_order": seasonal_order,
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q, "m": m,
            "aic": fit_result["aic"],
            "bic": fit_result["bic"],
            "success": fit_result["success"],
        })

        if (idx + 1) % 20 == 0:
            log.info(f"[{name}] Progress: {idx + 1}/{total} models tested")

    results_df = pd.DataFrame(results)
    successful = results_df[results_df["success"]]

    if len(successful) == 0:
        log.warning(f"[{name}] No successful SARIMA models!")
        return {
            "best_order": None,
            "best_seasonal_order": None,
            "all_results": results_df,
        }

    successful = successful.sort_values(criterion)
    best_row = successful.iloc[0]

    log.info(f"[{name}] Best: SARIMA{best_row['order']}x{best_row['seasonal_order']} "
             f"({criterion}={best_row[criterion]:.2f})")

    return {
        "best_order": best_row["order"],
        "best_seasonal_order": best_row["seasonal_order"],
        "best_aic": best_row["aic"],
        "best_bic": best_row["bic"],
        "all_results": results_df,
        "top_5": successful.head(5),
    }


# ── Model Diagnostics ──────────────────────────────────────────────────────────
def diagnose_residuals(
    residuals: pd.Series,
    name: str = "Model"
) -> dict:
    """
    Perform comprehensive residual diagnostics.

    Good model residuals should be:
    - Zero mean
    - Constant variance (homoscedastic)
    - Normally distributed
    - No autocorrelation (white noise)
    """
    resid = residuals.dropna()

    # Basic statistics
    mean_resid = resid.mean()
    std_resid = resid.std()
    skewness = resid.skew()
    kurtosis = resid.kurtosis()

    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(resid)
    is_normal = jb_pvalue > 0.05

    # Ljung-Box test for autocorrelation
    lb_result = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
    min_lb_pvalue = lb_result["lb_pvalue"].min()
    is_white_noise = min_lb_pvalue > 0.05

    results = {
        "mean": mean_resid,
        "std": std_resid,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "is_normal": is_normal,
        "ljung_box_results": lb_result,
        "min_ljung_box_pvalue": min_lb_pvalue,
        "is_white_noise": is_white_noise,
        "passes_diagnostics": is_normal and is_white_noise,
    }

    status_normal = "✓" if is_normal else "✗"
    status_wn = "✓" if is_white_noise else "✗"

    log.info(f"[{name}] Residual diagnostics:")
    log.info(f"  Mean: {mean_resid:.4f}, Std: {std_resid:.4f}")
    log.info(f"  Normality (JB): p={jb_pvalue:.4f} {status_normal}")
    log.info(f"  White Noise (LB): p={min_lb_pvalue:.4f} {status_wn}")

    return results


def plot_diagnostics(
    residuals: pd.Series,
    name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create diagnostic plots for model residuals.

    Standard diagnostic plots:
    1. Residuals over time
    2. Histogram + normal fit
    3. Q-Q plot
    4. ACF of residuals
    """
    resid = residuals.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} - Residual Diagnostics", fontsize=14, fontweight="bold")

    # Residuals over time
    ax = axes[0, 0]
    ax.plot(resid.index, resid.values, linewidth=0.8)
    ax.axhline(y=0, color="red", linestyle="--")
    ax.set_title("Residuals Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")

    # Histogram
    ax = axes[0, 1]
    ax.hist(resid, bins=50, density=True, edgecolor="black", alpha=0.7)
    # Fit normal distribution
    x = np.linspace(resid.min(), resid.max(), 100)
    ax.plot(x, stats.norm.pdf(x, resid.mean(), resid.std()),
            "r-", linewidth=2, label="Normal fit")
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend()

    # Q-Q plot
    ax = axes[1, 0]
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot (Normality Check)")

    # ACF of residuals
    ax = axes[1, 1]
    plot_acf(resid, lags=30, ax=ax, title="")
    ax.set_title("ACF of Residuals (White Noise Check)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved diagnostics plot → {save_path.name}")

    return fig


# ── Main Analysis Function ─────────────────────────────────────────────────────
def analyze_arima(
    series: pd.Series,
    name: str,
    test_size: int = 30,
    suggested_order: Optional[tuple] = None,
    do_grid_search: bool = True,
    save_plots: bool = True
) -> dict:
    """
    Complete ARIMA analysis for an asset.

    Parameters
    ----------
    series : pd.Series
        Price series
    name : str
        Asset name
    test_size : int
        Number of observations for testing
    suggested_order : tuple, optional
        Suggested (p,d,q) from ACF/PACF analysis
    do_grid_search : bool
        Whether to perform grid search
    save_plots : bool
        Whether to save plots

    Returns
    -------
    dict with all analysis results
    """
    log.info(f"\n{'='*60}")
    log.info(f"[{name}] ARIMA Analysis")
    log.info(f"{'='*60}")

    # Train/test split
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    horizon = len(test)

    log.info(f"[{name}] Train: {len(train)} obs, Test: {len(test)} obs")

    # Grid search if requested
    if do_grid_search:
        grid_result = grid_search_arima(
            train,
            p_range=range(0, 4),
            d_range=range(0, 3),
            q_range=range(0, 4),
            name=name
        )
        best_order = grid_result["best_order"]
        if best_order is None:
            best_order = suggested_order or (1, 1, 1)
    else:
        best_order = suggested_order or (1, 1, 1)
        grid_result = None

    # Fit best ARIMA model
    arima_result = arima_forecast(train, horizon, best_order, name=name)

    # Also try SARIMA
    seasonal_order = (1, 0, 1, 21)  # Monthly seasonality
    sarima_result = sarima_forecast(train, horizon, best_order, seasonal_order, name=name)

    # Evaluate
    metrics_arima = calculate_metrics(test.values, arima_result["forecast"].values)
    metrics_sarima = calculate_metrics(test.values, sarima_result["forecast"].values)

    arima_result["metrics"] = metrics_arima
    sarima_result["metrics"] = metrics_sarima

    # Diagnostics on best model
    if arima_result["success"]:
        diagnostics = diagnose_residuals(arima_result["resid"], f"{name} ARIMA{best_order}")
    else:
        diagnostics = None

    # Comparison
    comparison = pd.DataFrame([
        {
            "Model": f"ARIMA{best_order}",
            "RMSE": metrics_arima["rmse"],
            "MAE": metrics_arima["mae"],
            "AIC": arima_result.get("aic", np.nan),
            "BIC": arima_result.get("bic", np.nan),
        },
        {
            "Model": f"SARIMA{best_order}x{seasonal_order}",
            "RMSE": metrics_sarima["rmse"],
            "MAE": metrics_sarima["mae"],
            "AIC": sarima_result.get("aic", np.nan),
            "BIC": sarima_result.get("bic", np.nan),
        },
    ])
    comparison = comparison.sort_values("RMSE")

    # Summary
    summary = f"""
    ARIMA Analysis: {name}
    ─────────────────────────────
    Best ARIMA order: {best_order}
    ARIMA RMSE: {metrics_arima['rmse']:.2f}
    SARIMA RMSE: {metrics_sarima['rmse']:.2f}

    Comparison:
{comparison.to_string(index=False)}

    Residual Diagnostics:
    {f"Passes: {'Yes' if diagnostics and diagnostics['passes_diagnostics'] else 'No'}" if diagnostics else 'N/A'}
    """
    log.info(summary)

    # Save plots
    if save_plots:
        # Forecast plot
        plot_path = RESULTS_DIR / f"{name}_arima_forecast.png"
        plot_arima_forecast(train, test, arima_result, sarima_result, name, plot_path)
        plt.close()

        # Diagnostics plot
        if arima_result["success"]:
            diag_path = RESULTS_DIR / f"{name}_arima_diagnostics.png"
            plot_diagnostics(arima_result["resid"], f"{name} ARIMA{best_order}", diag_path)
            plt.close()

    return {
        "best_order": best_order,
        "arima_result": arima_result,
        "sarima_result": sarima_result,
        "comparison": comparison,
        "diagnostics": diagnostics,
        "grid_search": grid_result,
        "train": train,
        "test": test,
        "summary": summary,
    }


def plot_arima_forecast(
    train: pd.Series,
    test: pd.Series,
    arima_result: dict,
    sarima_result: dict,
    name: str,
    save_path: Optional[Path] = None,
    show_last_n: int = 100
) -> plt.Figure:
    """Plot ARIMA and SARIMA forecasts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{name} - ARIMA/SARIMA Forecasts", fontsize=14, fontweight="bold")

    train_plot = train.iloc[-show_last_n:]

    for ax, result, title in zip(
        axes,
        [arima_result, sarima_result],
        [arima_result.get("method", "ARIMA"), sarima_result.get("method", "SARIMA")]
    ):
        # Training data
        ax.plot(train_plot.index, train_plot.values, label="Training", color="gray", alpha=0.7)

        # Actual
        ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)

        # Forecast
        if result["success"]:
            ax.plot(result["forecast"].index, result["forecast"].values,
                    label="Forecast", color="blue", linewidth=2, linestyle="--")

            # Confidence interval
            ax.fill_between(result["forecast"].index,
                            result["lower"].values,
                            result["upper"].values,
                            alpha=0.2, color="blue", label="95% CI")

            metrics = result.get("metrics", {})
            ax.set_title(f"{title}\nRMSE={metrics.get('rmse', 0):.2f}, AIC={result.get('aic', 0):.0f}")
        else:
            ax.set_title(f"{title}\nFailed to fit")

        ax.legend(loc="upper left")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved forecast plot → {save_path.name}")

    return fig


def analyze_all_assets(test_size: int = 30, do_grid_search: bool = False) -> dict:
    """
    Run ARIMA analysis on all assets.

    Note: Set do_grid_search=False for faster execution.
    """
    log.info("=" * 60)
    log.info("ARIMA Analysis - All Assets")
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

        result = analyze_arima(
            series, name,
            test_size=test_size,
            do_grid_search=do_grid_search
        )
        results[name] = result

        for _, row in result["comparison"].iterrows():
            summary_rows.append({
                "Asset": name,
                "Model": row["Model"],
                "RMSE": row["RMSE"],
                "MAE": row["MAE"],
                "AIC": row["AIC"],
            })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "arima_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path.name}")

    # Best by asset
    best_by_asset = summary_df.loc[summary_df.groupby("Asset")["RMSE"].idxmin()]
    print("\n" + "=" * 60)
    print("ARIMA/SARIMA - BEST MODEL BY ASSET")
    print("=" * 60)
    print(best_by_asset[["Asset", "Model", "RMSE", "AIC"]].to_string(index=False))
    print("=" * 60)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Run with grid search disabled for speed
    # Set do_grid_search=True for full analysis
    results = analyze_all_assets(test_size=30, do_grid_search=False)
