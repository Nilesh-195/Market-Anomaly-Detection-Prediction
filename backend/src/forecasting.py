"""
forecasting.py
==============
Master forecasting module that integrates all forecasting methods.

Provides a unified interface for:
  1. Running complete forecasting analysis pipeline
  2. Comparing all methods side-by-side
  3. Generating comprehensive reports
  4. Selecting best method per asset

Methods included:
  - Naive (Mean, Naive, Seasonal Naive, Drift)
  - Exponential Smoothing (SES, Holt, Damped, Holt-Winters)
  - ARIMA/SARIMA
  - VAR (for multi-asset)

This is the main entry point for the TSFA course project deliverables.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all forecasting modules
from stationarity import analyze_stationarity, analyze_all_assets as analyze_stationarity_all
from acf_pacf_analysis import analyze_acf_pacf, analyze_all_assets as analyze_acf_pacf_all
from naive_methods import analyze_naive_methods, analyze_all_assets as analyze_naive_all
from exponential_smoothing import analyze_exp_smoothing, analyze_all_assets as analyze_exp_smoothing_all
from arima_models import analyze_arima, analyze_all_assets as analyze_arima_all
from var_model import analyze_var, analyze_all_combinations as analyze_var_all

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
RESULTS_DIR   = ROOT_DIR / "backend" / "results"
REPORT_DIR    = RESULTS_DIR / "forecast_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]


def run_full_pipeline(
    test_size: int = 30,
    do_grid_search: bool = False,
    save_plots: bool = True
) -> dict:
    """
    Run the complete forecasting pipeline for all assets.

    Pipeline steps:
    1. Stationarity analysis (ADF/KPSS tests)
    2. ACF/PACF analysis (ARIMA order selection)
    3. Naive methods (baseline)
    4. Exponential smoothing
    5. ARIMA/SARIMA
    6. VAR (multi-asset)
    7. Method comparison and best selection

    Parameters
    ----------
    test_size : int
        Number of observations for testing
    do_grid_search : bool
        Whether to perform ARIMA grid search (slower but more thorough)
    save_plots : bool
        Whether to save all plots

    Returns
    -------
    dict with all results organized by step
    """
    log.info("=" * 70)
    log.info("COMPLETE FORECASTING PIPELINE - TSFA COURSE PROJECT")
    log.info("=" * 70)

    results = {
        "stationarity": {},
        "acf_pacf": {},
        "naive": {},
        "exp_smoothing": {},
        "arima": {},
        "var": {},
        "comparison": {},
        "best_methods": {},
    }

    # Step 1: Stationarity Analysis
    log.info("\n" + "=" * 70)
    log.info("STEP 1: STATIONARITY ANALYSIS")
    log.info("=" * 70)
    results["stationarity"] = analyze_stationarity_all()

    # Get d values for ARIMA
    d_values = {}
    for name, result in results["stationarity"].items():
        d_values[name] = result["d"]

    # Step 2: ACF/PACF Analysis
    log.info("\n" + "=" * 70)
    log.info("STEP 2: ACF/PACF ANALYSIS")
    log.info("=" * 70)
    results["acf_pacf"] = analyze_acf_pacf_all(d_values=d_values)

    # Step 3: Naive Methods
    log.info("\n" + "=" * 70)
    log.info("STEP 3: NAIVE METHODS (BASELINE)")
    log.info("=" * 70)
    results["naive"] = analyze_naive_all(test_size=test_size)

    # Step 4: Exponential Smoothing
    log.info("\n" + "=" * 70)
    log.info("STEP 4: EXPONENTIAL SMOOTHING")
    log.info("=" * 70)
    results["exp_smoothing"] = analyze_exp_smoothing_all(test_size=test_size)

    # Step 5: ARIMA
    log.info("\n" + "=" * 70)
    log.info("STEP 5: ARIMA/SARIMA MODELS")
    log.info("=" * 70)
    results["arima"] = analyze_arima_all(test_size=test_size, do_grid_search=do_grid_search)

    # Step 6: VAR
    log.info("\n" + "=" * 70)
    log.info("STEP 6: VAR (MULTI-ASSET)")
    log.info("=" * 70)
    results["var"] = analyze_var_all(test_size=test_size)

    # Step 7: Comparison
    log.info("\n" + "=" * 70)
    log.info("STEP 7: METHOD COMPARISON & SELECTION")
    log.info("=" * 70)
    comparison, best_methods = compare_all_methods(results)
    results["comparison"] = comparison
    results["best_methods"] = best_methods

    # Generate report
    generate_report(results)

    log.info("\n" + "=" * 70)
    log.info("PIPELINE COMPLETE")
    log.info(f"Results saved to: {REPORT_DIR}")
    log.info("=" * 70)

    return results


def compare_all_methods(results: dict) -> tuple[pd.DataFrame, dict]:
    """
    Compare all forecasting methods across all assets.

    Returns
    -------
    tuple of (comparison DataFrame, dict of best method per asset)
    """
    comparison_rows = []

    for asset in ASSETS:
        # Naive methods
        if asset in results["naive"]:
            for _, row in results["naive"][asset]["comparison"].iterrows():
                comparison_rows.append({
                    "Asset": asset,
                    "Category": "Naive",
                    "Method": row["Method"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                })

        # Exponential smoothing
        if asset in results["exp_smoothing"]:
            for _, row in results["exp_smoothing"][asset]["comparison"].iterrows():
                comparison_rows.append({
                    "Asset": asset,
                    "Category": "Exp. Smoothing",
                    "Method": row["Method"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                })

        # ARIMA
        if asset in results["arima"]:
            for _, row in results["arima"][asset]["comparison"].iterrows():
                comparison_rows.append({
                    "Asset": asset,
                    "Category": "ARIMA",
                    "Method": row["Model"],
                    "RMSE": row["RMSE"],
                    "MAE": row["MAE"],
                })

    comparison_df = pd.DataFrame(comparison_rows)

    # Find best method per asset
    best_methods = {}
    for asset in ASSETS:
        asset_df = comparison_df[comparison_df["Asset"] == asset]
        if len(asset_df) > 0:
            best_idx = asset_df["RMSE"].idxmin()
            best_row = asset_df.loc[best_idx]
            best_methods[asset] = {
                "method": best_row["Method"],
                "category": best_row["Category"],
                "rmse": best_row["RMSE"],
                "mae": best_row["MAE"],
            }

    # Log comparison
    log.info("\nBEST METHOD BY ASSET:")
    for asset, best in best_methods.items():
        log.info(f"  {asset}: {best['method']} ({best['category']}) - RMSE={best['rmse']:.2f}")

    return comparison_df, best_methods


def generate_report(results: dict) -> None:
    """
    Generate comprehensive forecast report.
    """
    log.info("Generating forecast report...")

    # 1. Save comparison CSV
    comparison_path = REPORT_DIR / "method_comparison.csv"
    results["comparison"].to_csv(comparison_path, index=False)
    log.info(f"Saved: {comparison_path.name}")

    # 2. Save best methods JSON
    best_path = REPORT_DIR / "best_methods.json"
    with open(best_path, "w") as f:
        json.dump(results["best_methods"], f, indent=2)
    log.info(f"Saved: {best_path.name}")

    # 3. Generate summary report
    report_lines = [
        "=" * 70,
        "TIME SERIES FORECASTING ANALYSIS - SUMMARY REPORT",
        "=" * 70,
        "",
        "ASSETS ANALYZED:",
        f"  {', '.join(ASSETS)}",
        "",
        "METHODS COMPARED:",
        "  - Naive: Mean, Naive, Seasonal Naive, Drift",
        "  - Exponential Smoothing: SES, Holt, Damped, Holt-Winters",
        "  - ARIMA/SARIMA",
        "  - VAR (multi-asset)",
        "",
        "STATIONARITY ANALYSIS:",
    ]

    for asset, result in results["stationarity"].items():
        d = result["d"]
        report_lines.append(f"  {asset}: d={d} (differencing needed)")

    report_lines.extend([
        "",
        "ACF/PACF SUGGESTED ORDERS:",
    ])

    for asset, result in results["acf_pacf"].items():
        order = result["suggested_order"]
        report_lines.append(f"  {asset}: ARIMA{order}")

    report_lines.extend([
        "",
        "BEST FORECASTING METHOD BY ASSET:",
    ])

    for asset, best in results["best_methods"].items():
        report_lines.append(
            f"  {asset}: {best['method']} ({best['category']}) - RMSE={best['rmse']:.2f}"
        )

    report_lines.extend([
        "",
        "VAR MULTI-ASSET ANALYSIS:",
    ])

    for combo, result in results["var"].items():
        if result.get("success"):
            report_lines.append(
                f"  {combo}: Lag={result['var_result']['lag_order']}, "
                f"AIC={result['var_result']['aic']:.2f}"
            )

    report_lines.extend([
        "",
        "=" * 70,
        "Generated by TSFA Course Project Pipeline",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    report_path = REPORT_DIR / "forecast_summary.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info(f"Saved: {report_path.name}")

    # Print to console
    print("\n" + report_text)


def quick_forecast(
    asset: str,
    horizon: int = 30,
    method: str = "auto"
) -> dict:
    """
    Quick forecast for a single asset using specified method.

    Parameters
    ----------
    asset : str
        Asset name (e.g., 'SP500')
    horizon : int
        Forecast horizon in days
    method : str
        Method to use ('naive', 'ses', 'holt', 'arima', 'auto')

    Returns
    -------
    dict with forecast and metadata
    """
    from naive_methods import naive_forecast
    from exponential_smoothing import holt_forecast
    from arima_models import arima_forecast

    # Load data
    path = PROCESSED_DIR / f"{asset}_features.parquet"
    if not path.exists():
        return {"error": f"Asset {asset} not found"}

    df = pd.read_parquet(path)
    series = df["Close"]
    train = series.iloc[:-horizon] if horizon < len(series) else series

    if method == "naive":
        result = naive_forecast(train, horizon)
    elif method == "ses":
        from exponential_smoothing import ses_forecast
        result = ses_forecast(train, horizon)
    elif method == "holt":
        result = holt_forecast(train, horizon)
    elif method == "arima":
        result = arima_forecast(train, horizon, order=(1, 1, 1), name=asset)
    elif method == "auto":
        # Run comparison and pick best
        from naive_methods import run_all_naive_methods
        from exponential_smoothing import run_all_exp_smoothing

        test = series.iloc[-horizon:]
        naive_results = run_all_naive_methods(train, test, name=asset)
        exp_results = run_all_exp_smoothing(train, test, name=asset)

        # Compare
        best_naive = naive_results["comparison"].iloc[0]["RMSE"]
        best_exp = exp_results["comparison"].iloc[0]["RMSE"]

        if best_naive < best_exp:
            result = naive_results["methods"][naive_results["best_method"]]
        else:
            result = exp_results["methods"][exp_results["best_method"]]

    return {
        "asset": asset,
        "horizon": horizon,
        "method": method,
        "forecast": result["forecast"],
        "lower": result.get("lower"),
        "upper": result.get("upper"),
    }


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument("--test-size", type=int, default=30, help="Test set size")
    parser.add_argument("--grid-search", action="store_true", help="Enable ARIMA grid search")
    parser.add_argument("--quick", type=str, help="Quick forecast for single asset")

    args = parser.parse_args()

    if args.quick:
        result = quick_forecast(args.quick, horizon=args.test_size)
        print(f"\nForecast for {args.quick}:")
        print(result["forecast"])
    else:
        results = run_full_pipeline(
            test_size=args.test_size,
            do_grid_search=args.grid_search,
        )
