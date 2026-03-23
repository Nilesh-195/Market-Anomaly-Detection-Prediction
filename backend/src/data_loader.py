"""
data_loader.py
==============
Downloads 14 years of daily OHLCV data for 6 main assets + 5 macro assets via yfinance.
Saves each asset as a parquet file in backend/data/raw/.
Also creates crash_labels.json in backend/data/ with 25 labeled market events.

Main Assets:
  ^GSPC  — S&P 500 Index
  ^VIX   — CBOE Volatility Index (Fear Index)
  BTC-USD — Bitcoin / USD
  GLD    — SPDR Gold ETF
  QQQ    — Invesco Nasdaq-100 ETF
  TSLA   — Tesla Inc.

Macro Assets (Phase 1 addition):
  ^TNX    — 10-year Treasury Yield
  ^TYX    — 30-year Treasury Yield
  DX-Y.NYB — US Dollar Index
  HYG     — High Yield Corporate Bond ETF
  TLT     — 20+ Year Treasury Bond ETF
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]          # project root
RAW_DIR    = ROOT_DIR / "backend" / "data" / "raw"
DATA_DIR   = ROOT_DIR / "backend" / "data"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ASSETS = {
    "^GSPC":   "SP500",
    "^VIX":    "VIX",
    "BTC-USD": "BTC",
    "GLD":     "GOLD",
    "QQQ":     "NASDAQ",
    "TSLA":    "TESLA",
}

# ── Macro/Cross-Asset Tickers ──────────────────────────────────────────────────
# Leading indicators for market stress — signals that move before crashes
MACRO_ASSETS = {
    "^TNX":      "TNX",      # 10-year Treasury yield (yield curve)
    "^TYX":      "TYX",      # 30-year Treasury yield
    "DX-Y.NYB":  "DXY",      # US Dollar Index — spikes during risk-off
    "HYG":       "HYG",      # High yield bond ETF — credit spread proxy
    "TLT":       "TLT",      # Long bond ETF — flight to safety signal
}

START_DATE = "2010-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# ── Ground truth crash events ──────────────────────────────────────────────────
# 13 events spanning full data range 2010-2026 — no gap > 18 months
# Note: Lehman Brothers (2008-09-15) excluded — predates data range (2010-01-01)
CRASH_EVENTS = [
    {
        "date":            "2010-05-06",
        "event":           "Flash Crash",
        "description":     "Market dropped 9% in minutes then recovered same day. Algo-driven cascade failure.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "VIX"],
    },
    {
        "date":            "2011-08-08",
        "event":           "US Debt Downgrade",
        "description":     "S&P downgraded US credit rating from AAA to AA+. Dow dropped 634 pts in one session.",
        "impact":          "medium",
        "assets_affected": ["SP500", "NASDAQ", "GOLD", "VIX"],
    },
    {
        "date":            "2015-08-24",
        "event":           "China Black Monday",
        "description":     "Global markets crashed on China economic slowdown fears. Shanghai Composite dropped 8.5%.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "BTC", "VIX"],
    },
    {
        "date":            "2018-02-05",
        "event":           "Volmageddon",
        "description":     "VIX spiked 115% overnight. Short-volatility ETPs collapsed. Dow fell 1,175 pts.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "VIX"],
    },
    {
        "date":            "2018-12-24",
        "event":           "Christmas Eve Crash 2018",
        "description":     "S&P 500 fell nearly 20% from peak. Fed rate-hike fears, US government shutdown, trade war concerns.",
        "impact":          "medium",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "VIX"],
    },
    {
        "date":            "2020-02-24",
        "event":           "COVID-19 First Wave Selloff",
        "description":     "First major crash day of COVID panic. S&P fell 3.4% as virus spread outside China became clear.",
        "impact":          "extreme",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "BTC", "VIX"],
    },
    {
        "date":            "2020-03-16",
        "event":           "COVID-19 Crash Peak",
        "description":     "S&P 500 dropped 34% in 5 weeks. VIX reached all-time high of 82.69. Circuit breakers triggered.",
        "impact":          "extreme",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "BTC", "GOLD", "VIX"],
    },
    {
        "date":            "2021-01-27",
        "event":           "GameStop Short Squeeze",
        "description":     "Reddit WallStreetBets drove GME up 2,500% in days. Robinhood halted trading.",
        "impact":          "medium",
        "assets_affected": ["SP500", "VIX"],
    },
    {
        "date":            "2022-01-24",
        "event":           "Fed Tightening Selloff",
        "description":     "Aggressive Fed rate-hike signals. Nasdaq fell 15% in January. Growth stocks hardest hit.",
        "impact":          "medium",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "BTC", "VIX"],
    },
    {
        "date":            "2022-05-12",
        "event":           "Luna/Terra Collapse",
        "description":     "$40 billion in crypto wiped out in 72 hours. UST de-pegged. Contagion across crypto.",
        "impact":          "high",
        "assets_affected": ["BTC", "VIX"],
    },
    {
        "date":            "2022-09-26",
        "event":           "UK Gilt Crisis",
        "description":     "UK bond market meltdown after Kwarteng mini-budget. Pound hit all-time low vs USD.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "GOLD", "VIX"],
    },
    {
        "date":            "2023-03-10",
        "event":           "Silicon Valley Bank Collapse",
        "description":     "SVB failed in 48 hours — largest US bank failure since 2008. Emergency Fed backstop required.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "BTC", "GOLD", "VIX"],
    },
    {
        "date":            "2024-08-05",
        "event":           "Yen Carry Trade Unwind",
        "description":     "Bank of Japan rate hike triggered global carry trade unwind. VIX spiked to 65. Nikkei fell 12%.",
        "impact":          "extreme",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "BTC", "VIX"],
    },
    # ── Phase 1 Expansion: +12 new events (13 → 25 total) ──
    {
        "date":            "2011-07-01",
        "event":           "European Debt Crisis Peak",
        "description":     "Greece bailout fears. Italian and Spanish yields spiked. EURUSD collapsed.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "GOLD", "VIX"],
    },
    {
        "date":            "2013-06-20",
        "event":           "Taper Tantrum",
        "description":     "Bernanke hints at QE tapering. 10-year yield jumped 100bps in weeks.",
        "impact":          "medium",
        "assets_affected": ["SP500", "NASDAQ", "GOLD", "VIX"],
    },
    {
        "date":            "2014-10-15",
        "event":           "US Treasury Flash Rally",
        "description":     "10-year Treasury yield flash crashed. Equity vol spiked briefly.",
        "impact":          "medium",
        "assets_affected": ["SP500", "VIX"],
    },
    {
        "date":            "2016-06-24",
        "event":           "Brexit Vote",
        "description":     "UK voted to leave EU. GBP crashed 10%. Global markets fell sharply.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "GOLD", "VIX"],
    },
    {
        "date":            "2018-10-10",
        "event":           "October 2018 Selloff",
        "description":     "Tech selloff triggered by rate fears. Nasdaq fell 10% in 10 days.",
        "impact":          "medium",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "VIX"],
    },
    {
        "date":            "2019-09-17",
        "event":           "Repo Market Crisis",
        "description":     "Overnight repo rates spiked to 10%. Fed emergency repo injections required.",
        "impact":          "medium",
        "assets_affected": ["SP500", "VIX"],
    },
    {
        "date":            "2022-03-07",
        "event":           "Russia-Ukraine Commodity Shock",
        "description":     "Oil hit $130. Commodity prices spiked globally. Stagflation fears.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "GOLD", "VIX"],
    },
    {
        "date":            "2022-11-09",
        "event":           "FTX Collapse",
        "description":     "FTX crypto exchange collapsed. $8B shortfall. Bitcoin fell 25% in 48hrs.",
        "impact":          "high",
        "assets_affected": ["BTC", "VIX"],
    },
    {
        "date":            "2023-05-01",
        "event":           "First Republic Bank Failure",
        "description":     "First Republic Bank seized by FDIC. Third major US bank failure of 2023.",
        "impact":          "medium",
        "assets_affected": ["SP500", "NASDAQ", "VIX"],
    },
    {
        "date":            "2025-04-07",
        "event":           "US Tariff Shock 2025",
        "description":     "Trump tariff announcements triggered global selloff. S&P fell 10% in 3 days.",
        "impact":          "extreme",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "BTC", "VIX"],
    },
    {
        "date":            "2025-01-27",
        "event":           "DeepSeek AI Shock",
        "description":     "Chinese AI lab DeepSeek released R1. Nvidia fell 17% in one day. Tech selloff.",
        "impact":          "high",
        "assets_affected": ["SP500", "NASDAQ", "TESLA", "VIX"],
    },
]


# ── Core download function ─────────────────────────────────────────────────────
def download_asset(ticker: str, name: str) -> pd.DataFrame | None:
    """Download OHLCV data for one asset and save as parquet."""
    out_path = RAW_DIR / f"{name}.parquet"

    if out_path.exists():
        log.info(f"[{name}] Already exists — loading from disk.")
        return pd.read_parquet(out_path)

    log.info(f"[{name}] Downloading {ticker} from {START_DATE} to {END_DATE} …")
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            log.warning(f"[{name}] No data returned for {ticker}.")
            return None

        # Flatten multi-level columns if present (yfinance ≥ 0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Keep only standard OHLCV columns (VIX has no Volume — fill 0)
        needed = ["Open", "High", "Low", "Close", "Volume"]
        for col in needed:
            if col not in df.columns:
                df[col] = 0
        df = df[needed]
        df["Volume"] = df["Volume"].fillna(0)
        df = df.dropna(subset=["Close"])

        df.to_parquet(out_path)
        log.info(
            f"[{name}] Saved {len(df):,} rows  "
            f"({df.index[0].date()} → {df.index[-1].date()})  →  {out_path.name}"
        )
        return df

    except Exception as exc:
        log.error(f"[{name}] Download failed: {exc}")
        return None


def save_crash_labels() -> None:
    """Write ground_truth crash events to backend/data/crash_labels.json."""
    out_path = DATA_DIR / "crash_labels.json"
    payload = {
        "description": (
            "Ground-truth anomaly events used for model evaluation. "
            "Each date represents a known market crash or extreme event. "
            "Data range: 2010-01-01 to present. "
            "Phase 1: Expanded from 13 to 25 events for improved supervised learning."
        ),
        "total_events": len(CRASH_EVENTS),
        "coverage_note": "25 events spanning full data range 2010-2026 with comprehensive market coverage",
        "impact_scale": {
            "extreme": "Cross-market systemic event, >30% drawdown or VIX > 50",
            "high":    "Major multi-market selloff, VIX spike > 30%",
            "medium":  "Elevated volatility, sector or asset-class specific impact",
        },
        "events": CRASH_EVENTS,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Crash labels saved → {out_path.name}  ({len(CRASH_EVENTS)} events)")


def validate_data(name: str, df: pd.DataFrame) -> None:
    """Run basic quality checks and log a summary."""
    n_rows   = len(df)
    n_na     = df["Close"].isna().sum()
    n_zero   = (df["Close"] == 0).sum()
    date_min = df.index.min().date()
    date_max = df.index.max().date()
    years    = (df.index.max() - df.index.min()).days / 365.25

    log.info(
        f"[{name}] QA → rows={n_rows:,}  NaN={n_na}  zero={n_zero}  "
        f"range={date_min} → {date_max}  ({years:.1f} yrs)"
    )
    if n_na > 0:
        log.warning(f"[{name}] {n_na} NaN values in Close column.")
    if years < 10:
        log.warning(f"[{name}] Only {years:.1f} years of data (expected ≥10).")


def load_all_assets() -> dict[str, pd.DataFrame]:
    """Load all 6 assets from parquet (must already be downloaded)."""
    result = {}
    for ticker, name in ASSETS.items():
        path = RAW_DIR / f"{name}.parquet"
        if not path.exists():
            log.warning(f"[{name}] Parquet not found — run download_all() first.")
            continue
        df = pd.read_parquet(path)
        result[name] = df
        log.info(f"[{name}] Loaded {len(df):,} rows from disk.")
    return result


def download_all() -> dict[str, pd.DataFrame]:
    """Download all 6 assets, validate them, and save crash labels."""
    log.info("=" * 60)
    log.info(f"Market Anomaly — Data Collection")
    log.info(f"Period : {START_DATE} → {END_DATE}")
    log.info(f"Assets : {', '.join(ASSETS.values())}")
    log.info("=" * 60)

    datasets = {}
    for ticker, name in ASSETS.items():
        df = download_asset(ticker, name)
        if df is not None:
            validate_data(name, df)
            datasets[name] = df

    save_crash_labels()

    log.info("=" * 60)
    log.info(f"Download complete — {len(datasets)}/{len(ASSETS)} assets ready.")
    log.info("=" * 60)
    return datasets


def download_macro_assets() -> dict[str, pd.DataFrame]:
    """
    Download macro/sentiment data for cross-asset features.
    These are leading indicators that move before crashes happen.
    """
    log.info("=" * 60)
    log.info(f"Macro Assets — Leading Indicators")
    log.info(f"Period : {START_DATE} → {END_DATE}")
    log.info(f"Assets : {', '.join(MACRO_ASSETS.values())}")
    log.info("=" * 60)

    datasets = {}
    for ticker, name in MACRO_ASSETS.items():
        df = download_asset(ticker, name)
        if df is not None:
            validate_data(name, df)
            datasets[name] = df

    log.info("=" * 60)
    log.info(f"Macro download complete — {len(datasets)}/{len(MACRO_ASSETS)} assets ready.")
    log.info("=" * 60)
    return datasets


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_all()
