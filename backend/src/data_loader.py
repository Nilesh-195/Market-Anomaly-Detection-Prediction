"""
data_loader.py
==============
Downloads 14 years of daily OHLCV data for 6 assets via yfinance.
Saves each asset as a parquet file in backend/data/raw/.
Also creates crash_labels.json in backend/data/.

Assets:
  ^GSPC  — S&P 500 Index
  ^VIX   — CBOE Volatility Index (Fear Index)
  BTC-USD — Bitcoin / USD
  GLD    — SPDR Gold ETF
  QQQ    — Invesco Nasdaq-100 ETF
  TSLA   — Tesla Inc.
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

START_DATE = "2010-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

# ── Ground truth crash events ──────────────────────────────────────────────────
CRASH_EVENTS = [
    {
        "date":        "2010-05-06",
        "event":       "Flash Crash",
        "description": "Market dropped 9% in minutes then recovered same day. Algo-driven.",
        "impact":      "high",
    },
    {
        "date":        "2011-08-08",
        "event":       "US Debt Downgrade",
        "description": "S&P downgraded US credit rating. Dow dropped 634 pts.",
        "impact":      "medium",
    },
    {
        "date":        "2015-08-24",
        "event":       "China Black Monday",
        "description": "Global markets crashed on China economic slowdown fears.",
        "impact":      "high",
    },
    {
        "date":        "2018-02-05",
        "event":       "Volmageddon",
        "description": "VIX spiked 115% overnight. Volatility products collapsed.",
        "impact":      "high",
    },
    {
        "date":        "2018-12-24",
        "event":       "Christmas Crash 2018",
        "description": "S&P 500 fell nearly 20% from peak. Fed rate-hike fears.",
        "impact":      "medium",
    },
    {
        "date":        "2020-03-16",
        "event":       "COVID-19 Crash",
        "description": "S&P 500 dropped 34% in 5 weeks. Highest VIX reading ever recorded.",
        "impact":      "extreme",
    },
    {
        "date":        "2021-01-27",
        "event":       "GameStop Short Squeeze",
        "description": "Reddit-driven short squeeze sent GME up 2500% in days.",
        "impact":      "medium",
    },
    {
        "date":        "2022-01-24",
        "event":       "Fed Tightening Selloff",
        "description": "Markets dropped sharply on aggressive Fed rate-hike signals.",
        "impact":      "medium",
    },
    {
        "date":        "2022-05-12",
        "event":       "Luna/Terra Collapse",
        "description": "$40 billion crypto wiped out in 72 hours. Contagion across crypto.",
        "impact":      "high",
    },
    {
        "date":        "2022-09-26",
        "event":       "UK Gilt Crisis",
        "description": "UK bond market meltdown after mini-budget. Pound hit all-time low.",
        "impact":      "high",
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
            "Each date represents a known market crash or extreme event."
        ),
        "total_events": len(CRASH_EVENTS),
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


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_all()
