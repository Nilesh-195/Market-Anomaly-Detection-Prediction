"""
features.py
===========
Computes 25 features for every asset needed by all anomaly detection models.

Phase 1 Update: Base 15 features + 10 macro cross-asset features

Features built per asset
────────────────────────
Base Features (15)
──────────────────
Price / Return
  1.  log_return          — daily log return
  2.  zscore_10           — Z-score of log_return vs 10-day rolling mean/std
  3.  zscore_20           — Z-score of log_return vs 20-day rolling mean/std
  4.  zscore_60           — Z-score of log_return vs 60-day rolling mean/std

Volatility
  5.  vol_10              — 10-day rolling std of log returns (annualised)
  6.  vol_30              — 30-day rolling std of log returns (annualised)
  7.  vol_ratio           — vol_10 / vol_30  (short/long vol ratio)

Drawdown
  8.  drawdown            — % drawdown from rolling 252-day high

Trend / Bubble
  9.  bubble_score        — (Close - 200d SMA) / 200d SMA * 100

Technical Indicators (via `ta` library)
  10. rsi_14              — RSI 14-period
  11. bb_position         — Bollinger Band position: (Close - lower) / (upper - lower)
  12. macd_hist           — MACD histogram (12/26/9)

Volume
  13. volume_zscore       — Z-score of Volume vs 20-day rolling mean/std
  14. vwap_deviation      — % deviation of Close from VWAP (rolling 20-day)

Composite
  15. atr_ratio           — ATR(14) / Close  (normalised true range)

Macro Features (10) — Phase 1 Addition
───────────────────────────────────────
Leading indicators of market stress:
  16. yield_curve         — TNX - TYX spread (negative = inversion)
  17. yield_curve_change  — Rate of change in yield curve
  18. dxy_return          — Daily return of US Dollar Index
  19. dxy_zscore          — Z-score of DXY return
  20. credit_stress       — Inverse of HYG returns (credit spread proxy)
  21. vix_level           — Current VIX level (if not VIX asset)
  22. vix_change_5d       — 5-day % change in VIX
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend    import MACD
from ta.volatility import BollingerBands, AverageTrueRange

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
RAW_DIR       = ROOT_DIR / "backend" / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "backend" / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = ["SP500", "VIX", "BTC", "GOLD", "NASDAQ", "TESLA"]


def make_sequences(X: np.ndarray, window: int) -> np.ndarray:
    """Build sliding window sequences from feature array."""
    return np.array([X[i: i + window] for i in range(len(X) - window + 1)])


# ── Helper: rolling Z-score ───────────────────────────────────────────────────
def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / std.replace(0, np.nan)


# ── Helper: VWAP deviation ────────────────────────────────────────────────────
def _vwap_deviation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling VWAP using typical price × volume over `window` days."""
    typical  = (df["High"] + df["Low"] + df["Close"]) / 3
    vol      = df["Volume"].replace(0, np.nan)
    vwap     = (typical * vol).rolling(window).sum() / vol.rolling(window).sum()
    return ((df["Close"] - vwap) / vwap * 100).fillna(0)


# ── Core feature builder ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Takes raw OHLCV DataFrame, returns a new DataFrame with all 15+ features.
    Original OHLCV columns are preserved alongside features.
    """
    feat = df.copy()

    # ── 1. Log return ──────────────────────────────────────────────────────────
    feat["log_return"] = np.log(feat["Close"] / feat["Close"].shift(1))

    # ── 2-4. Rolling Z-scores ──────────────────────────────────────────────────
    feat["zscore_10"] = _rolling_zscore(feat["log_return"], 10)
    feat["zscore_20"] = _rolling_zscore(feat["log_return"], 20)
    feat["zscore_60"] = _rolling_zscore(feat["log_return"], 60)

    # ── 5-6. Rolling volatility (annualised %) ─────────────────────────────────
    feat["vol_10"] = feat["log_return"].rolling(10).std() * np.sqrt(252) * 100
    feat["vol_30"] = feat["log_return"].rolling(30).std() * np.sqrt(252) * 100

    # ── 7. Volatility ratio ────────────────────────────────────────────────────
    feat["vol_ratio"] = (feat["vol_10"] / feat["vol_30"].replace(0, np.nan)).fillna(1)

    # ── 8. Drawdown from 252-day high ──────────────────────────────────────────
    rolling_max       = feat["Close"].rolling(252, min_periods=1).max()
    feat["drawdown"]  = (feat["Close"] - rolling_max) / rolling_max * 100

    # ── 9. Bubble score (distance from 200-day SMA) ────────────────────────────
    sma_200             = feat["Close"].rolling(200, min_periods=100).mean()
    feat["bubble_score"] = ((feat["Close"] - sma_200) / sma_200 * 100).fillna(0)

    # ── 10. RSI (14) ───────────────────────────────────────────────────────────
    rsi              = RSIIndicator(close=feat["Close"], window=14)
    feat["rsi_14"]   = rsi.rsi()

    # ── 11. Bollinger Band position ────────────────────────────────────────────
    bb = BollingerBands(close=feat["Close"], window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_position"] = ((feat["Close"] - bb_lower) / band_width).fillna(0.5)

    # ── 12. MACD histogram ─────────────────────────────────────────────────────
    macd             = MACD(close=feat["Close"], window_slow=26,
                            window_fast=12, window_sign=9)
    feat["macd_hist"] = macd.macd_diff().fillna(0)

    # ── 13. Volume Z-score ─────────────────────────────────────────────────────
    # VIX has zero volume — keep feature but set to 0
    if feat["Volume"].sum() == 0:
        feat["volume_zscore"] = 0.0
    else:
        feat["volume_zscore"] = _rolling_zscore(feat["Volume"], 20).fillna(0)

    # ── 14. VWAP deviation ─────────────────────────────────────────────────────
    if feat["Volume"].sum() == 0:
        feat["vwap_deviation"] = 0.0
    else:
        feat["vwap_deviation"] = _vwap_deviation(feat, window=20)

    # ── 15. ATR ratio ──────────────────────────────────────────────────────────
    atr              = AverageTrueRange(high=feat["High"], low=feat["Low"],
                                        close=feat["Close"], window=14)
    feat["atr_ratio"] = (atr.average_true_range() / feat["Close"].replace(0, np.nan) * 100).fillna(0)

    # ── Drop first 200 rows (insufficient history for all rolling windows) ──────
    feat = feat.dropna(subset=["log_return", "rsi_14", "zscore_60"]).copy()

    # ── Feature column list (no OHLCV) ────────────────────────────────────────
    feature_cols = [
        "log_return",
        "zscore_10", "zscore_20", "zscore_60",
        "vol_10",    "vol_30",    "vol_ratio",
        "drawdown",  "bubble_score",
        "rsi_14",    "bb_position", "macd_hist",
        "volume_zscore", "vwap_deviation",
        "atr_ratio",
    ]

    log.info(
        f"[{name}] Features built → {len(feat):,} rows × {len(feature_cols)} features  "
        f"({feat.index[0].date()} → {feat.index[-1].date()})"
    )
    return feat[["Open", "High", "Low", "Close", "Volume"] + feature_cols]


def build_macro_features(df: pd.DataFrame, macro_data: dict, name: str) -> pd.DataFrame:
    """
    Adds 10 macro/cross-asset features to an existing feature DataFrame.
    Call this AFTER build_features(), pass the result of build_features() as df.

    Macro features are leading indicators of market stress — signals that move
    before crashes happen, not just after.

    New features:
      - yield_curve: TNX - TYX spread (negative = inversion, predicts recessions)
      - yield_curve_change: Rate of change in yield curve
      - dxy_return: Daily return of US Dollar Index (spikes during risk-off)
      - dxy_zscore: Z-score of DXY return (extreme = flight to dollar)
      - credit_stress: Inverse of HYG returns (HYG falls = credit spreads widen)
      - vix_level: Current VIX level (fear gauge)
      - vix_change_5d: 5-day % change in VIX (rapid spikes = panic)
    """
    feat = df.copy()

    # ── Yield curve (TNX - TYX spread) ──────────────────────────────────
    if 'TNX' in macro_data and 'TYX' in macro_data:
        tnx = macro_data['TNX']['Close'].reindex(feat.index, method='ffill')
        tyx = macro_data['TYX']['Close'].reindex(feat.index, method='ffill')
        feat['yield_curve'] = (tnx - tyx).fillna(0)         # negative = inversion
        feat['yield_curve_change'] = feat['yield_curve'].diff().fillna(0)

    # ── Dollar index (DXY) ───────────────────────────────────────────────
    if 'DXY' in macro_data:
        dxy = macro_data['DXY']['Close'].reindex(feat.index, method='ffill')
        dxy_ret = np.log(dxy / dxy.shift(1)).fillna(0)
        feat['dxy_return'] = dxy_ret
        feat['dxy_zscore'] = _rolling_zscore(dxy_ret, 20).fillna(0)

    # ── Credit spread proxy (HYG inverse) ───────────────────────────────
    if 'HYG' in macro_data:
        hyg = macro_data['HYG']['Close'].reindex(feat.index, method='ffill')
        hyg_ret = np.log(hyg / hyg.shift(1)).fillna(0)
        feat['credit_stress'] = (-hyg_ret).fillna(0)    # HYG falls = credit stress rises

    # ── VIX term structure (rolling 5-day change in VIX) ────────────────
    if 'VIX' in macro_data and name != 'VIX':
        vix = macro_data['VIX']['Close'].reindex(feat.index, method='ffill')
        feat['vix_level'] = vix.ffill().fillna(20)
        feat['vix_change_5d'] = vix.pct_change(5).fillna(0)

    macro_cols = [c for c in feat.columns if c not in df.columns]
    log.info(f'[{name}] Macro features added: {macro_cols} ({len(macro_cols)} new features)')
    return feat


# ── Validate: check for NaN / Inf ─────────────────────────────────────────────
def validate_features(df: pd.DataFrame, name: str) -> None:
    feature_cols = [c for c in df.columns if c not in ("Open","High","Low","Close","Volume")]
    n_nan = df[feature_cols].isna().sum().sum()
    n_inf = np.isinf(df[feature_cols].values).sum()
    if n_nan > 0:
        log.warning(f"[{name}] {n_nan} NaN values in feature columns.")
        per_col = df[feature_cols].isna().sum()
        log.warning(f"[{name}] NaN breakdown:\n{per_col[per_col>0]}")
    if n_inf > 0:
        log.warning(f"[{name}] {n_inf} Inf values detected — clipping.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    if n_nan == 0 and n_inf == 0:
        log.info(f"[{name}] ✅ All features valid — 0 NaN, 0 Inf.")


# ── Process all assets ────────────────────────────────────────────────────────
def build_all_features() -> dict[str, pd.DataFrame]:
    log.info("=" * 60)
    log.info("Feature Engineering — building 15 base + 10 macro features for 6 assets")
    log.info("=" * 60)

    # Load macro assets once (shared across all asset features)
    log.info("Loading macro assets for cross-asset features...")
    macro_data = {}
    macro_names = ['TNX', 'TYX', 'DXY', 'HYG', 'VIX']
    for macro_name in macro_names:
        macro_path = RAW_DIR / f'{macro_name}.parquet'
        if macro_path.exists():
            macro_data[macro_name] = pd.read_parquet(macro_path)
            log.info(f"  ✓ Loaded {macro_name} ({len(macro_data[macro_name]):,} rows)")
        else:
            log.warning(f"  ✗ {macro_name} not found — macro features may be incomplete")

    if not macro_data:
        log.warning("No macro assets found — skipping macro features. Run download_macro_assets() first.")

    results = {}
    for name in ASSETS:
        raw_path = RAW_DIR / f"{name}.parquet"
        if not raw_path.exists():
            log.warning(f"[{name}] Raw parquet not found — skipping.")
            continue

        df_raw  = pd.read_parquet(raw_path)
        df_raw.index = pd.to_datetime(df_raw.index)

        # Build base features (15 original features)
        df_feat = build_features(df_raw, name)

        # Add macro features if macro data is available
        if macro_data:
            df_feat = build_macro_features(df_feat, macro_data, name)

        validate_features(df_feat, name)

        out_path = PROCESSED_DIR / f"{name}_features.parquet"
        df_feat.to_parquet(out_path)
        log.info(f"[{name}] Saved → {out_path.name}")

        results[name] = df_feat

    log.info("=" * 60)
    log.info(f"Feature engineering complete — {len(results)}/6 assets processed.")
    log.info("=" * 60)
    return results


def load_all_features() -> dict[str, pd.DataFrame]:
    """Load all processed feature parquets from disk."""
    results = {}
    for name in ASSETS:
        path = PROCESSED_DIR / f"{name}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            results[name] = df
            log.info(f"[{name}] Loaded {len(df):,} rows × {len(df.columns)} cols from disk.")
        else:
            log.warning(f"[{name}] Features not found — run build_all_features() first.")
    return results


# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    datasets = build_all_features()

    # Quick summary
    print("\n── Feature Summary ──────────────────────────────────────────")
    for name, df in datasets.items():
        feat_cols = [c for c in df.columns if c not in ("Open","High","Low","Close","Volume")]
        print(f"  {name:7}  {len(df):5,} rows  {len(feat_cols)} features  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
    print("─" * 60)
