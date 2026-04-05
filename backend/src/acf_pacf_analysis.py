"""
acf_pacf_analysis.py
====================
Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis for ARIMA order selection.

Implements:
  - ACF computation and visualization
  - PACF computation and visualization
  - Automatic ARIMA (p,d,q) order recommendation
  - Ljung-Box test for residual autocorrelation

Theory:
  ACF (Autocorrelation Function):
    - Measures correlation between y_t and y_{t-k} for all lags k
    - Includes indirect effects through intermediate lags
    - Pattern helps identify MA order (q):
      • Sharp cutoff after lag q → MA(q) process
      • Slow decay → AR component present

  PACF (Partial Autocorrelation Function):
    - Measures direct correlation between y_t and y_{t-k}
    - Removes effects of intermediate lags
    - Pattern helps identify AR order (p):
      • Sharp cutoff after lag p → AR(p) process
      • Slow decay → MA component present

  ARIMA Order Selection Guide:
  ┌───────────────┬──────────────┬──────────────────┐
  │ ACF Pattern   │ PACF Pattern │ Model            │
  ├───────────────┼──────────────┼──────────────────┤
  │ Cuts off q    │ Tails off    │ MA(q)            │
  │ Tails off     │ Cuts off p   │ AR(p)            │
  │ Tails off     │ Tails off    │ ARMA(p,q)        │
  │ Cuts off q    │ Cuts off p   │ ARMA(p,q)        │
  └───────────────┴──────────────┴──────────────────┘
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
RESULTS_DIR   = ROOT_DIR / "backend" / "results" / "acf_pacf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── ACF Computation ────────────────────────────────────────────────────────────
def compute_acf(
    series: pd.Series,
    nlags: int = 40,
    alpha: float = 0.05
) -> dict:
    """
    Compute ACF values with confidence intervals.

    Parameters
    ----------
    series : pd.Series
        Stationary time series
    nlags : int
        Number of lags to compute
    alpha : float
        Significance level for confidence intervals (default 0.05 = 95% CI)

    Returns
    -------
    dict with keys:
        - acf_values: Array of ACF values
        - confint: Confidence interval bounds
        - nlags: Number of lags
        - significant_lags: Lags where ACF is significant
    """
    clean_series = series.dropna()

    acf_values, confint = acf(clean_series, nlags=nlags, alpha=alpha, fft=True)

    # Confidence bands (approx ±1.96/sqrt(n) for white noise)
    n = len(clean_series)
    upper_bound = 1.96 / np.sqrt(n)
    lower_bound = -1.96 / np.sqrt(n)

    # Find significant lags (excluding lag 0 which is always 1)
    significant_lags = []
    for i in range(1, len(acf_values)):
        if acf_values[i] > upper_bound or acf_values[i] < lower_bound:
            significant_lags.append(i)

    return {
        "acf_values": acf_values,
        "confint": confint,
        "nlags": nlags,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "significant_lags": significant_lags,
    }


def compute_pacf(
    series: pd.Series,
    nlags: int = 40,
    method: str = "ywm"
) -> dict:
    """
    Compute PACF values with confidence intervals.

    Parameters
    ----------
    series : pd.Series
        Stationary time series
    nlags : int
        Number of lags to compute
    method : str
        Method for computation ('ywm' = Yule-Walker modified)

    Returns
    -------
    dict with keys:
        - pacf_values: Array of PACF values
        - nlags: Number of lags
        - significant_lags: Lags where PACF is significant
    """
    clean_series = series.dropna()

    # Ensure nlags doesn't exceed valid range
    max_lags = min(nlags, len(clean_series) // 2 - 1)
    pacf_values = pacf(clean_series, nlags=max_lags, method=method)

    n = len(clean_series)
    upper_bound = 1.96 / np.sqrt(n)
    lower_bound = -1.96 / np.sqrt(n)

    # Find significant lags (excluding lag 0)
    significant_lags = []
    for i in range(1, len(pacf_values)):
        if pacf_values[i] > upper_bound or pacf_values[i] < lower_bound:
            significant_lags.append(i)

    return {
        "pacf_values": pacf_values,
        "nlags": max_lags,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "significant_lags": significant_lags,
    }


# ── ARIMA Order Suggestion ─────────────────────────────────────────────────────
def suggest_arima_order(
    series: pd.Series,
    d: int = 1,
    max_p: int = 5,
    max_q: int = 5,
    name: str = "Series"
) -> dict:
    """
    Suggest ARIMA(p,d,q) orders based on ACF/PACF analysis.

    Uses heuristics:
    - p = last significant lag in PACF before cutoff
    - q = last significant lag in ACF before cutoff
    - d = provided (should be from stationarity analysis)

    Parameters
    ----------
    series : pd.Series
        Original (or differenced) time series
    d : int
        Order of differencing (from stationarity analysis)
    max_p : int
        Maximum AR order to consider
    max_q : int
        Maximum MA order to consider
    name : str
        Name for logging

    Returns
    -------
    dict with:
        - suggested_p: Recommended AR order
        - suggested_d: Given differencing order
        - suggested_q: Recommended MA order
        - acf_analysis: Full ACF results
        - pacf_analysis: Full PACF results
        - interpretation: Text explanation
    """
    # Compute ACF and PACF
    acf_result = compute_acf(series, nlags=max(max_p, max_q) + 5)
    pacf_result = compute_pacf(series, nlags=max(max_p, max_q) + 5)

    # Determine p from PACF
    pacf_significant = [lag for lag in pacf_result["significant_lags"] if lag <= max_p]
    if pacf_significant:
        # Look for cutoff pattern - find where significance drops
        suggested_p = min(pacf_significant[-1], max_p)
    else:
        suggested_p = 0

    # Determine q from ACF
    acf_significant = [lag for lag in acf_result["significant_lags"] if lag <= max_q]
    if acf_significant:
        suggested_q = min(acf_significant[-1], max_q)
    else:
        suggested_q = 0

    # Build interpretation
    interpretation_lines = [
        f"ACF Analysis:",
        f"  • Significant lags: {acf_significant[:10] if acf_significant else 'None'}",
        f"  • Pattern suggests MA({suggested_q})",
        f"",
        f"PACF Analysis:",
        f"  • Significant lags: {pacf_significant[:10] if pacf_significant else 'None'}",
        f"  • Pattern suggests AR({suggested_p})",
        f"",
        f"Recommendation:",
        f"  • ARIMA({suggested_p},{d},{suggested_q})",
        f"",
        f"Alternative orders to try:",
    ]

    # Add alternative orders
    alternatives = []
    for p in range(max(0, suggested_p - 1), min(suggested_p + 2, max_p + 1)):
        for q in range(max(0, suggested_q - 1), min(suggested_q + 2, max_q + 1)):
            if (p, d, q) != (suggested_p, d, suggested_q):
                alternatives.append(f"ARIMA({p},{d},{q})")

    interpretation_lines.append(f"  • {', '.join(alternatives[:4])}")

    interpretation = "\n".join(interpretation_lines)

    log.info(f"[{name}] Suggested order: ARIMA({suggested_p},{d},{suggested_q})")
    log.info(f"[{name}] ACF significant lags: {acf_significant[:5]}")
    log.info(f"[{name}] PACF significant lags: {pacf_significant[:5]}")

    return {
        "suggested_p": suggested_p,
        "suggested_d": d,
        "suggested_q": suggested_q,
        "acf_analysis": acf_result,
        "pacf_analysis": pacf_result,
        "interpretation": interpretation,
        "alternatives": alternatives[:4],
    }


# ── Ljung-Box Test ─────────────────────────────────────────────────────────────
def ljung_box_test(
    residuals: pd.Series,
    lags: list[int] = [10, 20, 30],
    name: str = "Residuals"
) -> dict:
    """
    Perform Ljung-Box test on residuals to check for autocorrelation.

    H0: Residuals are independently distributed (no autocorrelation)
    H1: Residuals exhibit autocorrelation

    Good model residuals should have p-value > 0.05 (fail to reject H0).

    Parameters
    ----------
    residuals : pd.Series
        Model residuals to test
    lags : list[int]
        Lags to test
    name : str
        Name for logging

    Returns
    -------
    dict with:
        - test_results: DataFrame with lb_stat, lb_pvalue for each lag
        - is_white_noise: Boolean (True if p > 0.05 at all lags)
        - interpretation: Text explanation
    """
    clean_residuals = residuals.dropna()

    lb_result = acorr_ljungbox(clean_residuals, lags=lags, return_df=True)

    # Check if all p-values are > 0.05
    min_pvalue = lb_result["lb_pvalue"].min()
    is_white_noise = min_pvalue > 0.05

    if is_white_noise:
        interpretation = f"✓ Residuals appear to be white noise (min p-value = {min_pvalue:.4f} > 0.05)"
    else:
        failed_lags = lb_result[lb_result["lb_pvalue"] <= 0.05].index.tolist()
        interpretation = f"✗ Residuals show autocorrelation at lags: {failed_lags}"

    log.info(f"[{name}] Ljung-Box: {interpretation}")

    return {
        "test_results": lb_result,
        "is_white_noise": is_white_noise,
        "min_pvalue": min_pvalue,
        "interpretation": interpretation,
    }


# ── Visualization ──────────────────────────────────────────────────────────────
def plot_acf_pacf(
    series: pd.Series,
    name: str,
    nlags: int = 40,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create ACF and PACF plots for a time series.
    """
    clean_series = series.dropna()
    nlags = min(nlags, len(clean_series) // 2 - 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{name} - ACF/PACF Analysis", fontsize=14, fontweight="bold")

    # Top-left: Time series
    ax = axes[0, 0]
    ax.plot(clean_series.index, clean_series.values, linewidth=0.8)
    ax.set_title("Time Series (Input)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")

    # Top-right: Distribution
    ax = axes[0, 1]
    ax.hist(clean_series, bins=50, edgecolor="black", alpha=0.7)
    ax.set_title("Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    # Bottom-left: ACF
    ax = axes[1, 0]
    plot_acf(clean_series, lags=nlags, ax=ax, title="")
    ax.set_title("ACF (Autocorrelation Function)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")

    # Bottom-right: PACF
    ax = axes[1, 1]
    plot_pacf(clean_series, lags=nlags, ax=ax, method="ywm", title="")
    ax.set_title("PACF (Partial Autocorrelation Function)")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Partial Autocorrelation")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved plot → {save_path.name}")

    return fig


def plot_detailed_acf_pacf(
    original: pd.Series,
    differenced: pd.Series,
    d: int,
    name: str,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create detailed ACF/PACF comparison: original vs differenced.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"{name} - ACF/PACF Analysis (Original vs Differenced)", fontsize=14, fontweight="bold")

    nlags_orig = min(40, len(original.dropna()) // 2 - 1)
    nlags_diff = min(40, len(differenced.dropna()) // 2 - 1)

    # Row 1: Original series
    ax = axes[0, 0]
    ax.plot(original.index, original.values, linewidth=0.8)
    ax.set_title("Original Series")

    ax = axes[0, 1]
    plot_acf(original.dropna(), lags=nlags_orig, ax=ax, title="")
    ax.set_title("ACF (Original)")

    ax = axes[0, 2]
    plot_pacf(original.dropna(), lags=nlags_orig, ax=ax, method="ywm", title="")
    ax.set_title("PACF (Original)")

    # Row 2: Differenced series
    ax = axes[1, 0]
    ax.plot(differenced.index, differenced.values, linewidth=0.8, color="green")
    ax.set_title(f"Differenced Series (d={d})")

    ax = axes[1, 1]
    plot_acf(differenced.dropna(), lags=nlags_diff, ax=ax, title="")
    ax.set_title("ACF (Differenced)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    ax = axes[1, 2]
    plot_pacf(differenced.dropna(), lags=nlags_diff, ax=ax, method="ywm", title="")
    ax.set_title("PACF (Differenced)")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"[{name}] Saved plot → {save_path.name}")

    return fig


# ── Main Analysis Function ─────────────────────────────────────────────────────
def analyze_acf_pacf(
    series: pd.Series,
    name: str,
    d: int = 1,
    save_plots: bool = True
) -> dict:
    """
    Complete ACF/PACF analysis for ARIMA order selection.

    Parameters
    ----------
    series : pd.Series
        Original price series
    name : str
        Asset name
    d : int
        Differencing order (from stationarity analysis)
    save_plots : bool
        Whether to save plots to disk

    Returns
    -------
    dict containing analysis results and ARIMA order suggestions
    """
    log.info(f"\n{'='*60}")
    log.info(f"[{name}] ACF/PACF Analysis")
    log.info(f"{'='*60}")

    # Apply differencing
    differenced = series.copy()
    for _ in range(d):
        differenced = differenced.diff()
    differenced = differenced.dropna()

    # Get order suggestion
    order_suggestion = suggest_arima_order(differenced, d=d, name=name)

    # Generate summary
    p, q = order_suggestion["suggested_p"], order_suggestion["suggested_q"]
    summary = f"""
    ACF/PACF Analysis: {name}
    ─────────────────────────────
    Differencing applied: d={d}

    ACF (for MA order q):
      • Significant lags: {order_suggestion['acf_analysis']['significant_lags'][:5]}

    PACF (for AR order p):
      • Significant lags: {order_suggestion['pacf_analysis']['significant_lags'][:5]}

    Recommended ARIMA Order:
      ★ ARIMA({p},{d},{q})

    Alternative orders to try:
      • {', '.join(order_suggestion['alternatives'])}
    """

    # Save plots
    if save_plots:
        plot_path = RESULTS_DIR / f"{name}_acf_pacf.png"
        plot_detailed_acf_pacf(series, differenced, d, name, plot_path)
        plt.close()

    return {
        "original_series": series,
        "differenced_series": differenced,
        "d": d,
        "suggested_p": p,
        "suggested_q": q,
        "suggested_order": (p, d, q),
        "acf_analysis": order_suggestion["acf_analysis"],
        "pacf_analysis": order_suggestion["pacf_analysis"],
        "alternatives": order_suggestion["alternatives"],
        "summary": summary,
    }


def analyze_all_assets(d_values: Optional[dict] = None) -> dict:
    """
    Run ACF/PACF analysis on all 6 assets.

    Parameters
    ----------
    d_values : dict, optional
        Pre-computed differencing orders from stationarity analysis.
        If None, defaults to d=1 for all assets.
    """
    log.info("=" * 60)
    log.info("ACF/PACF Analysis - All Assets")
    log.info("=" * 60)

    # Default d values (prices are typically non-stationary)
    if d_values is None:
        d_values = {name: 1 for name in ASSETS}

    results = {}
    summary_rows = []

    for name in ASSETS:
        path = PROCESSED_DIR / f"{name}_features.parquet"
        if not path.exists():
            log.warning(f"[{name}] Features not found — skipping.")
            continue

        df = pd.read_parquet(path)
        series = df["Close"]
        d = d_values.get(name, 1)

        result = analyze_acf_pacf(series, name, d=d)
        results[name] = result

        summary_rows.append({
            "Asset": name,
            "d": d,
            "Suggested_p": result["suggested_p"],
            "Suggested_q": result["suggested_q"],
            "Suggested_Order": f"ARIMA({result['suggested_p']},{d},{result['suggested_q']})",
            "ACF_Significant": str(result["acf_analysis"]["significant_lags"][:3]),
            "PACF_Significant": str(result["pacf_analysis"]["significant_lags"][:3]),
        })
        log.info(result["summary"])

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "acf_pacf_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Summary saved → {summary_path.name}")

    # Print summary
    print("\n" + "=" * 60)
    print("ACF/PACF ANALYSIS SUMMARY - ARIMA ORDER RECOMMENDATIONS")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = analyze_all_assets()
