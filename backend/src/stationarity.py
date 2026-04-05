"""
stationarity.py
===============
Stationarity testing and transformation for time series forecasting.

Implements:
  - ADF (Augmented Dickey-Fuller) test for unit root detection
  - KPSS test as complementary stationarity check
  - Automatic differencing to achieve stationarity
  - Visualization of original vs differenced series

Theory:
  A stationary series has constant mean, variance, and autocorrelation over time.
  Non-stationary series must be differenced before ARIMA modeling.

  ADF Test:
    H0: Series has unit root (non-stationary)
    H1: Series is stationary
    Decision: Reject H0 if p-value < 0.05 → series is stationary

  KPSS Test:
    H0: Series is stationary (opposite of ADF)
    H1: Series has unit root
    Decision: Reject H0 if p-value < 0.05 → series is non-stationary
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
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
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "stationarity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── ADF Test ───────────────────────────────────────────────────────────────────
def adf_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Perform Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    series : pd.Series
        Time series to test (must not contain NaN)
    name : str
        Name for logging purposes

    Returns
    -------
    dict with keys:
        - adf_statistic: Test statistic
        - p_value: P-value (reject H0 if < 0.05)
        - critical_values: Dict of critical values at 1%, 5%, 10%
        - is_stationary: Boolean
        - n_lags: Number of lags used
        - n_obs: Number of observations
    """
    # Remove NaN
    clean_series = series.dropna()

    if len(clean_series) < 20:
        log.warning(f"[{name}] Series too short for ADF test ({len(clean_series)} obs)")
        return {
            "adf_statistic": np.nan,
            "p_value": np.nan,
            "critical_values": {},
            "is_stationary": False,
            "n_lags": 0,
            "n_obs": len(clean_series),
        }

    result = adfuller(clean_series, autolag="AIC")

    output = {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }

    status = "✓ Stationary" if output["is_stationary"] else "✗ Non-stationary"
    log.info(f"[{name}] ADF Test: stat={output['adf_statistic']:.4f}, "
             f"p-value={output['p_value']:.4f} → {status}")

    return output


def kpss_test(series: pd.Series, name: str = "Series", regression: str = "c") -> dict:
    """
    Perform KPSS test for stationarity.

    Note: KPSS has opposite null hypothesis to ADF.
    - ADF H0: Unit root exists (non-stationary)
    - KPSS H0: Series is stationary

    Parameters
    ----------
    series : pd.Series
        Time series to test
    name : str
        Name for logging
    regression : str
        'c' for constant (level stationarity), 'ct' for constant + trend

    Returns
    -------
    dict with keys:
        - kpss_statistic: Test statistic
        - p_value: P-value (reject H0 if < 0.05 → non-stationary)
        - critical_values: Dict of critical values
        - is_stationary: Boolean (True if we fail to reject H0)
        - n_lags: Number of lags used
    """
    clean_series = series.dropna()

    if len(clean_series) < 20:
        log.warning(f"[{name}] Series too short for KPSS test ({len(clean_series)} obs)")
        return {
            "kpss_statistic": np.nan,
            "p_value": np.nan,
            "critical_values": {},
            "is_stationary": False,
            "n_lags": 0,
        }

    result = kpss(clean_series, regression=regression, nlags="auto")

    output = {
        "kpss_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "critical_values": result[3],
        # KPSS: low p-value = reject H0 (stationary) → non-stationary
        "is_stationary": result[1] >= 0.05,
    }

    status = "✓ Stationary" if output["is_stationary"] else "✗ Non-stationary"
    log.info(f"[{name}] KPSS Test: stat={output['kpss_statistic']:.4f}, "
             f"p-value={output['p_value']:.4f} → {status}")

    return output


def comprehensive_stationarity_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Run both ADF and KPSS tests and provide combined interpretation.

    Interpretation Matrix:
    ┌─────────────┬─────────────┬─────────────────────────────┐
    │ ADF Result  │ KPSS Result │ Interpretation              │
    ├─────────────┼─────────────┼─────────────────────────────┤
    │ Stationary  │ Stationary  │ Stationary ✓               │
    │ Stationary  │ Non-stat    │ Trend-stationary (diff)    │
    │ Non-stat    │ Stationary  │ Difference-stationary      │
    │ Non-stat    │ Non-stat    │ Non-stationary (diff ≥1)   │
    └─────────────┴─────────────┴─────────────────────────────┘
    """
    adf_result = adf_test(series, name)
    kpss_result = kpss_test(series, name)

    adf_stationary = adf_result["is_stationary"]
    kpss_stationary = kpss_result["is_stationary"]

    if adf_stationary and kpss_stationary:
        interpretation = "stationary"
        recommendation = "No differencing needed (d=0)"
    elif adf_stationary and not kpss_stationary:
        interpretation = "trend_stationary"
        recommendation = "Consider detrending or d=1"
    elif not adf_stationary and kpss_stationary:
        interpretation = "difference_stationary"
        recommendation = "Apply differencing (d=1)"
    else:
        interpretation = "non_stationary"
        recommendation = "Apply differencing (d=1 or d=2)"

    log.info(f"[{name}] Combined: {interpretation} → {recommendation}")

    return {
        "adf": adf_result,
        "kpss": kpss_result,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "needs_differencing": interpretation != "stationary",
    }


# ── Differencing ───────────────────────────────────────────────────────────────
def difference_series(
    series: pd.Series,
    d: int = 1,
    name: str = "Series"
) -> pd.Series:
    """
    Apply differencing to a time series.

    Parameters
    ----------
    series : pd.Series
        Original time series
    d : int
        Order of differencing (1 = first difference, 2 = second difference)
    name : str
        Name for logging

    Returns
    -------
    pd.Series
        Differenced series (will have d fewer observations)
    """
    result = series.copy()
    for i in range(d):
        result = result.diff()

    result = result.dropna()
    log.info(f"[{name}] Applied d={d} differencing: {len(series)} → {len(result)} observations")
    return result


def auto_difference(
    series: pd.Series,
    name: str = "Series",
    max_d: int = 2
) -> tuple[pd.Series, int]:
    """
    Automatically determine and apply optimal differencing order.

    Starts with d=0, increments until stationarity is achieved or max_d is reached.

    Parameters
    ----------
    series : pd.Series
        Original time series
    name : str
        Name for logging
    max_d : int
        Maximum differencing order to try

    Returns
    -------
    tuple of (differenced_series, d_used)
    """
    log.info(f"[{name}] Auto-differencing: testing d=0 to d={max_d}")

    current = series.dropna()

    for d in range(max_d + 1):
        if d > 0:
            current = current.diff().dropna()

        adf_result = adf_test(current, f"{name} (d={d})")

        if adf_result["is_stationary"]:
            log.info(f"[{name}] ✓ Achieved stationarity at d={d}")
            return current, d

    log.warning(f"[{name}] Could not achieve stationarity with d≤{max_d}")
    return current, max_d


# ── Rolling Statistics for Visual Verification ─────────────────────────────────
def compute_rolling_stats(
    series: pd.Series,
    window: int = 60
) -> tuple[pd.Series, pd.Series]:
    """
    Compute rolling mean and std for visual stationarity check.

    A stationary series should have roughly constant rolling mean and std.
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return rolling_mean, rolling_std


# ── Visualization ──────────────────────────────────────────────────────────────
def plot_stationarity_analysis(
    original: pd.Series,
    differenced: pd.Series,
    d: int,
    name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create comprehensive stationarity visualization.

    Shows:
    - Original series with rolling mean/std
    - Differenced series with rolling mean/std
    - ADF test results for both
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} - Stationarity Analysis", fontsize=14, fontweight="bold")

    # Top-left: Original series
    ax = axes[0, 0]
    ax.plot(original.index, original.values, label="Original", alpha=0.8)
    roll_mean, roll_std = compute_rolling_stats(original)
    ax.plot(original.index, roll_mean, label="Rolling Mean (60d)", color="red", linewidth=2)
    ax.fill_between(original.index, roll_mean - roll_std, roll_mean + roll_std,
                    alpha=0.2, color="red", label="Rolling Std")
    ax.set_title("Original Series")
    ax.legend(loc="upper left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    # Top-right: Histogram of original
    ax = axes[0, 1]
    ax.hist(original.dropna(), bins=50, edgecolor="black", alpha=0.7)
    adf_orig = adf_test(original, f"{name}_orig")
    ax.set_title(f"Original Distribution\nADF p-value: {adf_orig['p_value']:.4f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    # Bottom-left: Differenced series
    ax = axes[1, 0]
    ax.plot(differenced.index, differenced.values, label=f"Differenced (d={d})", alpha=0.8, color="green")
    roll_mean, roll_std = compute_rolling_stats(differenced)
    ax.plot(differenced.index, roll_mean, label="Rolling Mean (60d)", color="red", linewidth=2)
    ax.fill_between(differenced.index, roll_mean - roll_std, roll_mean + roll_std,
                    alpha=0.2, color="red", label="Rolling Std")
    ax.set_title(f"Differenced Series (d={d})")
    ax.legend(loc="upper left")
    ax.set_xlabel("Date")
    ax.set_ylabel("Differenced Value")

    # Bottom-right: Histogram of differenced
    ax = axes[1, 1]
    ax.hist(differenced.dropna(), bins=50, edgecolor="black", alpha=0.7, color="green")
    adf_diff = adf_test(differenced, f"{name}_diff")
    ax.set_title(f"Differenced Distribution\nADF p-value: {adf_diff['p_value']:.4f}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved plot → {save_path.name}")

    return fig


# ── Main Analysis Function ─────────────────────────────────────────────────────
def analyze_stationarity(
    series: pd.Series,
    name: str,
    column: str = "Close",
    save_plots: bool = True
) -> dict:
    """
    Complete stationarity analysis for a time series.

    Parameters
    ----------
    series : pd.Series
        Price series to analyze
    name : str
        Asset name for logging and file naming
    column : str
        Column name (for logging)
    save_plots : bool
        Whether to save plots to disk

    Returns
    -------
    dict containing:
        - original_test: ADF/KPSS results for original series
        - differenced: The differenced series
        - d: Order of differencing used
        - differenced_test: ADF/KPSS results for differenced series
        - summary: Text summary
    """
    log.info(f"\n{'='*60}")
    log.info(f"[{name}] Stationarity Analysis")
    log.info(f"{'='*60}")

    # Test original series
    original_test = comprehensive_stationarity_test(series, f"{name} Original")

    # Auto-difference
    differenced, d = auto_difference(series, name)

    # Test differenced series
    if d > 0:
        diff_test = comprehensive_stationarity_test(differenced, f"{name} (d={d})")
    else:
        diff_test = original_test

    # Generate summary
    summary = f"""
    Stationarity Analysis: {name}
    ─────────────────────────────
    Original Series:
      • ADF statistic: {original_test['adf']['adf_statistic']:.4f}
      • ADF p-value:   {original_test['adf']['p_value']:.4f}
      • Conclusion:    {'Stationary' if original_test['adf']['is_stationary'] else 'Non-Stationary'}

    After Differencing (d={d}):
      • ADF statistic: {diff_test['adf']['adf_statistic']:.4f}
      • ADF p-value:   {diff_test['adf']['p_value']:.4f}
      • Conclusion:    {'Stationary' if diff_test['adf']['is_stationary'] else 'Non-Stationary'}

    Recommendation:
      Use d={d} for ARIMA modeling
    """

    # Save plot
    if save_plots:
        plot_path = RESULTS_DIR / f"{name}_stationarity.png"
        plot_stationarity_analysis(series, differenced, d, name, plot_path)
        plt.close()

    return {
        "original_test": original_test,
        "differenced": differenced,
        "d": d,
        "differenced_test": diff_test,
        "summary": summary,
    }


def analyze_all_assets() -> dict:
    """
    Run stationarity analysis on all 6 assets.
    """
    log.info("=" * 60)
    log.info("Stationarity Analysis - All Assets")
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

        result = analyze_stationarity(series, name)
        results[name] = result

        summary_rows.append({
            "Asset": name,
            "Original_ADF_pvalue": result["original_test"]["adf"]["p_value"],
            "Original_Stationary": result["original_test"]["adf"]["is_stationary"],
            "d_used": result["d"],
            "Final_ADF_pvalue": result["differenced_test"]["adf"]["p_value"],
            "Final_Stationary": result["differenced_test"]["adf"]["is_stationary"],
        })
        log.info(result["summary"])

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "stationarity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path.name}")

    # Print summary table
    print("\n" + "=" * 60)
    print("STATIONARITY ANALYSIS SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = analyze_all_assets()
