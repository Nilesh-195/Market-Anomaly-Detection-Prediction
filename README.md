# Market Anomaly Detection & Prediction

> **Production-grade ML system for stock price forecasting and market anomaly detection across 6 financial assets, powered by a 9-model ensemble with multi-mode anomaly forecasting, regime-adaptive blending, integrated explainability, REST API, and interactive React dashboard.** ⭐

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite&logoColor=white)](https://vite.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API Version](https://img.shields.io/badge/API-v2.1.0-brightgreen)](http://localhost:8000/docs)

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **Assets Covered** | 6 financial assets (S&P 500, VIX, Bitcoin, Gold, NASDAQ, Tesla) |
| **Historical Data** | 2010–2026 (~4,000 trading days per asset) |
| **Labelled Events** | 24 major market crashes with ±7 day detection windows |
| **Anomaly Detection** | 9 models (4 baseline + 5 advanced) + 2-tier ensemble approach |
| **Anomaly Forecasting Modes** ⭐ | 4 modes (Ensemble, Advanced, DL Composite, Hybrid) × 3 methods (ARIMA, ETS, Hybrid blend) |
| **Forecasting Methods** | 10+ approaches: naive baselines, ARIMA/SARIMA, VAR, deep learning (LSTM, Transformer), XGBoost |
| **Ensemble Performance** | 0.74 average ROC-AUC (0.84 S&P 500, 0.83 VIX, 0.80 NASDAQ) |
| **Detection Accuracy** | 38% average crash detection rate (67% Tesla, 54% NASDAQ, 46% VIX) |
| **Training Time** | ~74 seconds (Apple Silicon MPS) |
| **API Endpoints** | 20+ REST endpoints with comprehensive documentation |
| **User Interface** | 6-page React dashboard with multi-asset support and light/dark themes |

---

## Contents

- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Machine Learning Models](#machine-learning-models)
- [Forecasting Methods](#forecasting-methods)
- [Feature Engineering](#feature-engineering)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Multi-Mode Forecasting & Blending Logic](#multi-mode-forecasting--blending-logic-) ⭐
- [Labelled Events](#labelled-events)
- [Model Performance](#model-performance)
- [Technology Stack](#technology-stack)
- [Risk Scoring](#risk-scoring)

---

## System Architecture

### System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🎨  PRESENTATION LAYER                             ║
║                   React 19 Dashboard  ·  Vite 7  ·  Recharts              ║
║                                                                             ║
║  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐        ║
║  │Dashboard │ │ Forecast │ │Historical│ │  Regime  │ │ Advanced │        ║
║  │(Overview)│ │  (Price) │ │(Crashes) │ │Detection │ │ Anomaly  │        ║
║  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘        ║
║  • 6 asset selector cards          • Light / Dark theme toggle            ║
║  • Live ensemble risk gauge        • Auto-refresh every 5 minutes         ║
║  • Interactive Brush zoom          • Toast notification system            ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                        HTTP / JSON / CORS
                                     │
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🚀  API LAYER  ·  FastAPI 0.135  ·  Uvicorn             ║
║                        Port 8000  ·  Auto-Docs /docs                      ║
║                                                                             ║
║  Forecasting Group              │  Anomaly Detection Group                 ║
║  ──────────────────────────── │  ────────────────────────────             ║
║  /forecast/price/{asset}        │  /anomaly/current/{asset}               ║
║  /forecast/arima/{asset}        │  /anomaly/advanced/{asset}              ║
║  /forecast/naive/{asset}        │  /anomaly/regime/{asset}                ║
║  /forecast/exponential/{asset}  │  /anomaly/historical/{asset}            ║
║  /forecast/var                  │  /anomaly/compare-tiers/{asset}         ║
║  /forecast/compare/{asset}      │  /anomaly/forecast/{asset}              ║
║  /forecast/stationarity/{asset} │  /anomaly/comparison/{asset}            ║
║  /forecast/acf-pacf/{asset}     │                                         ║
║  /forecast/evaluate/{asset}     │  General: /  /assets  /summary          ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🧠  INFERENCE LAYER  ·  Model Registry                  ║
║                                                                             ║
║  ── ANOMALY DETECTION MODELS (per asset × 6) ───────────────────────────  ║
║  Baseline (4): Z-Score · Isolation Forest · LSTM Autoencoder · Prophet    ║
║  Advanced (5): XGBoost · HMM · TCN · VAE · Anomaly Transformer            ║
║  Ensembles:    Baseline Ensemble (4-model) · Advanced Ensemble (9-model)   ║
║                                                                             ║
║  ── FORECASTING MODELS ─────────────────────────────────────────────────  ║
║  Naive / Drift / Mean  │  SES / Holt / Holt-Winters  │  ARIMA / SARIMA   ║
║  VAR (multi-asset)     │  LSTM Seq2Seq (PyTorch)      │  Transformer       ║
║  XGBoost Regressor     │                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                 🗄️  DATA LAYER  ·  Feature Store                           ║
║                                                                             ║
║  yfinance OHLCV (2010 → 2026)  →  15 Engineered Features  →  Parquet      ║
║  Crash Labels JSON (24 events)  →  ±7 day detection window                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Ensemble Scoring

```
Baseline Ensemble Score = (15% × Z-Score)
                        + (25% × Isolation Forest)
                        + (40% × LSTM Autoencoder)
                        + (20% × Prophet Residual)

Advanced Ensemble Score  = Dynamic weighted average of all 9 models

Final Score Range: 0 – 100  (normalized)

  0 – 39   →  🟢 Normal       (no significant anomaly)
 40 – 59   →  🟡 Elevated     (monitor positions closely)
 60 – 74   →  🟠 High Risk    (review exposure & hedge)
 75 – 100  →  🔴 Extreme      (crisis-level alert)
```

---

## Core Features

**Anomaly Detection**
- Dual-tier ensemble: 4-model baseline (Z-Score, Isolation Forest, LSTM AE, Prophet) + 5-model advanced (XGBoost, HMM, TCN, VAE, Anomaly Transformer)
- Market regime classification (bull/bear/crisis) via HMM
- 24 labelled crash events with ±7 trading day detection window
- SHAP model interpretability and baseline/advanced comparison

**Multi-Mode Anomaly Forecasting** ⭐ *NEW*
- 4 distinct signal sources with toggleable UI modes:
  - **Ensemble:** Baseline 4-model average (Z-Score, IForest, LSTM AE, Prophet)
  - **Advanced:** 9-model weighted combining all baseline + advanced models
  - **DL Composite:** Deep learning weighted blend (0.35×LSTM + 0.30×TCN + 0.20×VAE + 0.15×Anomaly Transformer)
  - **Hybrid:** Regime-adaptive blend of Advanced + DL (crisis: 45/55, bull: 60/40, neutral: 55/45)
- 3 forecasting methods: ARIMA(2,1,2), ExponentialSmoothing (ETS), Hybrid blend (60% ARIMA + 40% ETS)
- Mode comparison mini-chart: Overlay all 4 forecasts to visualize signal divergence
- Per-forecast metadata: model sources and weights used

**Price Forecasting**
- Multi-method ranking: naive baselines → ARIMA/SARIMA → deep learning (LSTM Seq2Seq, Transformer)
- Automatic method selection via cross-validated RMSE
- 95% confidence intervals on all forecasts
- Stationarity testing, ACF/PACF analysis, VAR multi-asset modeling

**User Interface**
- Interactive React dashboard (5 pages, light/dark themes)
- Vertically stacked layout for better readability on all screen sizes
- Recharts visualizations (zoom, brush, tooltips, multi-mode overlay)
- Real-time updates with 5-minute auto-refresh
- Toast notifications and fully responsive design

---

## Machine Learning Models

### Anomaly Detection — Baseline Tier (4 models)

| Model | Library | Architecture | Key Parameters |
|-------|---------|--------------|-----------------|
| **Z-Score** | NumPy | Rolling window standardization | 20-day window on log-returns |
| **Isolation Forest** | scikit-learn | Tree-based anomaly isolation | 200 trees, 3% contamination rate, 15 features |
| **LSTM Autoencoder** | PyTorch 2.10 | Encoder 15→64→8, Decoder 8→64→15 | 30-day window, MSE reconstruction loss |
| **Prophet** | Facebook Prophet | Time series decomposition + residuals | Pre-2020 training, threshold = 3σ |

### Anomaly Detection — Advanced Tier (5 models)

| Model | Library | Architecture | Key Parameters |
|-------|---------|--------------|-----------------|
| **XGBoost Classifier** | xgboost 2.1 | Gradient boosting on features + labels | Trained on 24 crash events, 15 features |
| **Hidden Markov Model** | hmmlearn | 3-state regime classifier | Bull/Bear/Crisis states, Gaussian emissions |
| **Temporal CNN** | PyTorch | Dilated causal convolutions | 30-day receptive field, 1D filters |
| **Variational Autoencoder** | PyTorch | LSTM encoder-decoder with KL loss | Latent dimension 8, reconstruction scoring |
| **Anomaly Transformer** | PyTorch | ICLR 2022 architecture | Association discrepancy loss, 64-dim embedding |

### Price Forecasting Models

| Category | Method | Implementation | Features |
|----------|--------|----------------|----------|
| **Naive Baselines** | Mean, Naïve, Seasonal Naïve, Drift | Custom NumPy/Pandas | Fast benchmark methods |
| **Exponential Smoothing** | SES, Holt's Linear, Damped Trend, Holt-Winters | statsmodels | Adaptive trend and seasonality |
| **ARIMA Family** | ARIMA, SARIMA | statsmodels with auto_arima | Automatic order selection (p, d, q) |
| **Multivariate** | Vector Autoregression (VAR) | statsmodels | 3-asset co-movement, Granger causality |
| **Deep Learning** | LSTM Seq2Seq | PyTorch 2.10 | Encoder-decoder, 28 multivariate features |
| **Deep Learning** | Transformer | PyTorch 2.10 | Multi-head attention, positional encoding |
| **Gradient Boosting** | XGBoost Regressor | xgboost 2.1 | Lagged features with cyclic encodings |

> **Note:** Deep learning models augment these 15 features with the ensemble anomaly score to create a 28-dimensional input vector that jointly models market structure and anomaly dynamics.

---

## Feature Engineering

15 hand-crafted features per asset capture market microstructure, trend, volatility, and risk dynamics:

| Feature | Category | Description |
|---------|----------|-------------|
| `log_return` | Returns | Daily logarithmic return |
| `zscore_10`, `zscore_20`, `zscore_60` | Statistical | Rolling Z-score normalization (10, 20, 60-day windows) |
| `vol_10`, `vol_30` | Volatility | Rolling annualized volatility |
| `vol_ratio` | Volatility | Short-term / long-term volatility ratio |
| `drawdown` | Risk | Maximum 252-day drawdown |
| `bubble_score` | Trend | Deviation of price from 200-day SMA |
| `rsi_14` | Technical | Relative Strength Index (normalized to [0,1]) |
| `bb_position` | Technical | Bollinger Bands position indicator |
| `macd_hist` | Technical | MACD histogram (normalized) |
| `volume_zscore` | Volume | Volume anomaly (20-day Z-score) |
| `vwap_deviation` | Volume | Price deviation from Volume-Weighted Average Price |
| `atr_ratio` | Volatility | Average True Range to price ratio |

**Multivariate extension:** Deep learning models augment these 15 features with the ensemble anomaly score (28-dimensional vector) to jointly model market structure and anomaly dynamics.

---

## Project Structure

```
Market_Anomaly_Detection_Prediction/
├── backend/
│   ├── api/main.py                          # FastAPI server (20+ endpoints)
│   ├── data/{raw,processed}                 # OHLCV + engineered features
│   ├── models/
│   │   ├── evaluation_{report.json,summary.csv}
│   │   └── {SP500,NASDAQ,BTC,GOLD,VIX,TESLA}/  # 9 models per asset
│   ├── notebooks/01_eda.ipynb .. 03_train_and_evaluate.ipynb ★
│   ├── src/
│   │   ├── data_loader.py, features.py      # Data pipeline
│   │   ├── models.py, advanced_models.py    # Anomaly detection
│   │   ├── {vae_model,anomaly_transformer}.py  # Deep learning
│   │   ├── {dl_models,gb_models}.py         # Forecasting
│   │   ├── {arima_models,var_model}.py      # Time series
│   │   └── {stationarity,acf_pacf_analysis,predict}.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx, main.jsx                # Root entry
│   │   ├── index.css                        # Tokensystem
│   │   ├── pages/                           # Dashboard, Forecast, Historical, Regime, AdvancedAnomaly
│   │   ├── components/                      # Layout, UI, charts
│   │   ├── hooks/                           # useMarketData, useAutoRefresh
│   │   ├── services/, constants/, utils/    # API, config, helpers
│   │   └── assets/                          # Icons, images
│   └── {package.json,vite.config.js,tailwind.config.js}
├── logs/training.log
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11+, Node.js 18+, Git

### Installation & Setup

**1. Clone and configure environment**
```bash
git clone https://github.com/Nilesh-195/Market-Anomaly-Detection-Prediction.git
cd Market-Anomaly-Detection-Prediction

python -m venv venv
source venv/bin/activate              # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

**2. Prepare data**
```bash
python backend/src/data_loader.py      # Download 6 assets
python backend/src/features.py          # Engineer 15 features
```

**3. Train models**
```bash
jupyter notebook backend/notebooks/03_train_and_evaluate.ipynb
```
Execute: data loading → anomaly training → DL forecasting → evaluation → results

**4. Launch API**
```bash
uvicorn backend.api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

**5. Start frontend**
```bash
cd frontend && npm install && npm run dev
# Dashboard: http://localhost:5173
```

---

## API Reference

### General Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + API info |
| `GET` | `/assets` | List all 6 supported assets |
| `GET` | `/summary` | Current price, forecast & anomaly score for all assets |

### Forecasting Endpoints

#### Price Forecasting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/forecast/price/{asset}?horizon=30&method=auto` | **Recommended** — best-performing method with 95% CI |
| `GET` | `/forecast/naive/{asset}?horizon=30` | All 4 naive baseline methods |
| `GET` | `/forecast/exponential/{asset}?horizon=30&method=auto` | SES / Holt / Holt-Winters family |
| `GET` | `/forecast/arima/{asset}?horizon=30&p=1&d=1&q=1&seasonal=false` | ARIMA or SARIMA with custom order |
| `GET` | `/forecast/var?assets=SP500,NASDAQ,VIX&horizon=30` | Multi-asset vector autoregression |
| `GET` | `/forecast/compare/{asset}?horizon=30` | All methods ranked by RMSE |
| `GET` | `/forecast/stationarity/{asset}` | ADF and KPSS test results |
| `GET` | `/forecast/acf-pacf/{asset}?max_lags=40` | ACF/PACF plots with ARIMA order recommendation |
| `GET` | `/forecast/evaluate/{asset}` | Cross-validation metrics (RMSE, MAE, MAPE) |

#### Anomaly Forecasting (Multi-Mode) ⭐ *NEW*

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/anomaly/forecast/{asset}?days=10&mode=hybrid&method=hybrid` | **Multi-mode anomaly score forecast with 95% CI.** Query parameters: `mode` ∈ {ensemble, advanced, dl, hybrid}, `method` ∈ {arima, ets, hybrid}. Default: mode=hybrid, method=hybrid. Returns forecast_points array with score, lower/upper bounds, risk_label, model_used, source_label |

### Anomaly Detection Endpoints (Real-time)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/anomaly/current/{asset}` | Latest baseline ensemble score with per-model breakdown |
| `GET` | `/anomaly/advanced/{asset}` | All 9 models + both ensemble scores |
| `GET` | `/anomaly/regime/{asset}` | HMM market regime timeline (bull/bear/crisis) |
| `GET` | `/anomaly/historical/{asset}` | Historical anomaly events (clustered) |
| `GET` | `/anomaly/comparison/{asset}` | Per-model statistics and correlation matrix |
| `GET` | `/anomaly/compare-tiers/{asset}` | Baseline vs advanced ensemble comparison |

### Example Requests

```bash
# Get best 30-day price forecast for S&P 500
curl "http://localhost:8000/forecast/price/SP500?horizon=30"
```

```json
{
  "asset": "SP500",
  "current_price": 5843.00,
  "horizon": 30,
  "method": "arima",
  "model_info": { "aic": 49231.4, "category": "arima" },
  "forecast": {
    "values": [5851.2, 5857.4, "..."],
    "lower_95": [5790.0, 5795.2, "..."],
    "upper_95": [5912.4, 5919.6, "..."],
    "dates": ["2026-03-25", "2026-03-26", "..."]
  },
  "summary": {
    "forecast_30d": 5980.1,
    "expected_return_pct": 2.35
  }
}
```

```bash
# Get 10-day hybrid (regime-aware) anomaly forecast for S&P 500 with hybrid method
curl "http://localhost:8000/anomaly/forecast/SP500?days=10&mode=hybrid&method=hybrid"
```

```json
{
  "asset": "SP500",
  "forecast_mode": "hybrid",
  "forecast_method": "hybrid",
  "source_label": "Regime-aware blend (crisis: 45% Adv + 55% DL, bull: 60% Adv + 40% DL, else: 55% Adv + 45% DL)",
  "model_used": ["advanced_ensemble", "dl_composite"],
  "forecast_points": [
    {
      "date": "2026-04-08",
      "score": 42.3,
      "lower_95": 38.1,
      "upper_95": 46.5,
      "risk_label": "Elevated"
    },
    {
      "date": "2026-04-09",
      "score": 38.9,
      "lower_95": 34.2,
      "upper_95": 43.6,
      "risk_label": "Normal"
    }
  ]
}
```

```bash
# Get 10-day DL composite mode forecast for Bitcoin with ARIMA method
curl "http://localhost:8000/anomaly/forecast/BTC?days=10&mode=dl&method=arima"
```

```json
{
  "asset": "BTC",
  "forecast_mode": "dl",
  "forecast_method": "arima",
  "source_label": "DL Composite: 35% LSTM + 30% TCN + 20% VAE + 15% Anomaly Transformer",
  "model_used": ["lstm_autoencoder", "tcn_model", "vae_model", "anomaly_transformer"],
  "forecast_points": [
    {
      "date": "2026-04-08",
      "score": 52.1,
      "lower_95": 48.3,
      "upper_95": 55.9,
      "risk_label": "Elevated"
    }
  ]
}
```

```bash
# Get 10-day advanced ensemble mode forecast for NASDAQ with ETS method
curl "http://localhost:8000/anomaly/forecast/NASDAQ?days=10&mode=advanced&method=ets"
```

```json
{
  "asset": "NASDAQ",
  "forecast_mode": "advanced",
  "forecast_method": "ets",
  "source_label": "Advanced Ensemble: All 9 models (4 baseline + 5 advanced)",
  "model_used": ["zscore", "iforest", "lstm", "prophet", "xgb", "hmm", "tcn", "vae", "anomaly_transformer"],
  "forecast_points": [
    {
      "date": "2026-04-08",
      "score": 45.7,
      "lower_95": 40.2,
      "upper_95": 51.2,
      "risk_label": "Elevated"
    }
  ]
}
```

```bash
# Get advanced anomaly score for Bitcoin (all 9 models)
curl http://localhost:8000/anomaly/advanced/BTC
```

```json
{
  "asset": "BTC",
  "date": "2026-04-04",
  "adv_ensemble_score": 58.3,
  "risk_label": "Elevated",
  "model_scores": {
    "zscore": 18.2, "iforest": 35.6,
    "lstm": 52.3,   "prophet": 38.9,
    "xgb": 61.1,    "hmm": 44.0,
    "tcn": 55.7,    "vae": 49.2,
    "at": 67.4
  }
}
```

---

## Labelled Events

The following 24 major market events have been manually labelled and form the ground-truth dataset for model evaluation:

| Date | Event | Classification |
|------|-------|--------|
| 2010-05-06 | **Flash Crash** | High |
| 2011-07-01 | **European Debt Crisis Peak** | High |
| 2011-08-08 | **US Debt Downgrade** | Medium |
| 2013-06-20 | **Taper Tantrum** | Medium |
| 2014-10-15 | **US Treasury Flash Rally** | Medium |
| 2015-08-24 | **China Black Monday** | High |
| 2016-06-24 | **Brexit Vote** | High |
| 2018-02-05 | **Volmageddon** | High |
| 2018-10-10 | **October 2018 Selloff** | Medium |
| 2018-12-24 | **Christmas Eve Crash** | Medium |
| 2019-09-17 | **Repo Market Crisis** | Medium |
| 2020-02-24 | **COVID-19 First Wave** | Extreme |
| 2020-03-16 | **COVID-19 Crash Peak** | Extreme |
| 2021-01-27 | **GameStop Short Squeeze** | Medium |
| 2022-01-24 | **Fed Tightening Selloff** | Medium |
| 2022-03-07 | **Russia-Ukraine Commodity Shock** | High |
| 2022-05-12 | **Luna / Terra Collapse** | High |
| 2022-09-26 | **UK Gilt Crisis** | High |
| 2022-11-09 | **FTX Collapse** | High |
| 2023-03-10 | **Silicon Valley Bank Collapse** | High |
| 2023-05-01 | **First Republic Bank Failure** | Medium |
| 2024-08-05 | **Yen Carry Trade Unwind** | Extreme |
| 2025-01-27 | **DeepSeek AI Shock** | High |
| 2025-04-07 | **US Tariff Shock** | Extreme |

**Classification levels:**
- **Extreme:** Systemic crisis with >30% drawdown or >30% VIX spike
- **High:** Significant market disruption with elevated volatility
- **Medium:** Noticeable market anomaly with temporary impact

---

## Model Performance

The ensemble models were evaluated against all 24 labelled crash events using a ±7 trading-day detection window. ROC-AUC and F1 metrics were computed to assess discrimination and detection accuracy.

| Asset | Ensemble ROC-AUC | Hit Rate | Events Detected |
|-------|:---------------:|:-------:|:---------------:|
| S&P 500 | **0.841** | 23% | 3 / 13 |
| VIX | **0.832** | 46% | 6 / 13 |
| Bitcoin | 0.633 | 36% | 4 / 11 |
| Gold | 0.607 | 0% | 0 / 13 |
| NASDAQ | **0.803** | 54% | 7 / 13 |
| Tesla | 0.705 | 67% | 8 / 12 |
| **Mean** | **0.737** | **38%** | — |

> **Note:** Strong AUC scores (0.74 avg) confirm the ensemble correctly *ranks* anomalous days above normal ones without labelled supervision. GOLD's 0% hit-rate reflects its safe-haven nature — it typically rises during crash events rather than falling, so anomaly signals invert. Full per-model metrics are in `backend/models/evaluation_report.json` and `evaluation_summary.csv`.

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Data Ingestion** | `yfinance`, `pandas`, `pyarrow` | Latest |
| **Feature Engineering** | `ta` (technical analysis), `numpy`, `scipy` | Latest |
| **Baseline Anomaly** | `scikit-learn` (Isolation Forest), `PyTorch` (LSTM AE), `prophet` | 1.8 / 2.10 / 1.3 |
| **Advanced Anomaly** | `xgboost`, `hmmlearn`, `shap` | 2.1 / 0.3 / 0.46 |
| **DL Anomaly** | `PyTorch` (VAE, Anomaly Transformer, TCN) | 2.10 |
| **Forecasting** | `statsmodels` (ARIMA/VAR), `PyTorch` (LSTM Seq2Seq / Transformer) | 0.14 / 2.10 |
| **Notebooks** | `jupyter`, `ipykernel` | Latest |
| **Backend** | `FastAPI`, `uvicorn`, `pydantic` | 0.135 / 0.41 / 2.12 |
| **Frontend** | `React 19`, `Vite 7` | 19 / 7 |
| **Charts** | `Recharts` (with Brush zoom) | Latest |
| **Icons** | `lucide-react` | Latest |

---

## Multi-Mode Forecasting & Blending Logic ⭐ *NEW*

### Mode Definitions & Mathematical Formulas

#### 1. **Ensemble Mode** (Baseline)
Uses the 4-model baseline ensemble score:
```
ensemble_score = (15% × Z-Score) + (25% × IForest) + (40% × LSTM AE) + (20% × Prophet)
```
**Use case:** Conservative baseline; good for first-pass anomaly detection.

#### 2. **Advanced Mode**
Uses all 9 anomaly models weighted by a pre-computed ensemble in the data layer:
```
adv_ensemble_score = weighted_average(all 9 models)
where models = [zscore, iforest, lstm, prophet, xgb, hmm, tcn, vae, anomaly_transformer]
```
**Use case:** Comprehensive signal incorporating advanced models; recommended for institutional risk monitoring.

#### 3. **DL Composite Mode**
Weighted average of deep learning models only:
```
dl_composite = (0.35×LSTM + 0.30×TCN + 0.20×VAE + 0.15×AT) / Σ available_weights
```
Where:
- LSTM = LSTM Autoencoder reconstruction score
- TCN = Temporal Convolutional Network score
- VAE = Variational Autoencoder score
- AT = Anomaly Transformer score
- Σ available_weights = sum of weights for models with valid data (enables graceful degradation)

**Use case:** Pure deep learning signal; emphasizes learned hierarchical patterns.

#### 4. **Hybrid Mode** (Regime-Aware) ⭐ *RECOMMENDED*
Adaptive blend of Advanced + DL Composite based on HMM market regime:
```
if HMM_state == "crisis":
    hybrid = 0.45×advanced + 0.55×dl_composite
elif HMM_state == "bull":
    hybrid = 0.60×advanced + 0.40×dl_composite
else:  # bear or neutral
    hybrid = 0.55×advanced + 0.45×dl_composite

hybrid_final = clip(hybrid, 0, 100)
```
**Regime weights:**
- **Crisis:** Increase DL weight (55%) as deep learning captures systemic risk better
- **Bull:** Decrease DL weight (40%), rely more on advanced ensemble (60%)
- **Bear/Neutral:** Balanced 55/45 split

**Use case:** Production-grade signal with adaptive sensitivity to market regimes.

### Forecasting Methods

Each mode forecast is computed using one of 3 methods:

#### 1. **ARIMA Method**
AutoRegressive Integrated Moving Average with automatic order detection:
```
ARIMA(2,1,2) fitted on last 252 trading days with differencing
Forecast: ŷₜ = μ + Σ φᵢ(yₜ₋ᵢ) + Σ θⱼ(εₜ₋ⱼ)
```
**Provides:** Mean forecast + 95% CI via bootstrap resampling.

#### 2. **ETS Method**
Exponential Smoothing with additive trend:
```
ETS(additive, trend=additive, seasonal=None)
Smoothed_t = α·signal + (1-α)·(smoothed_{t-1} + trend_{t-1})
```
**Provides:** Trend-adjusted forecast + variance-based CI.

#### 3. **Hybrid Method** (60% ARIMA + 40% ETS)
Blended forecast combining both methods:
```
hybrid_forecast = 0.60×arima_forecast + 0.40×ets_forecast
hybrid_ci = sqrt(0.60²×arima_var + 0.40²×ets_var)
```
**Use case:** Robust forecast averaging technical strength of both approaches.

---

## Risk Scoring

The anomaly score ranges from 0 to 100 and should be interpreted according to the following framework. Scores are independent of forecasting mode/method but risk labels adjust based on regime:

| Score Range | Status | Interpretation | Recommended Action |
|:----------:|--------|-----------------|-------------------|
| 0–39 | Normal | No significant anomaly detected | Hold current positions |
| 40–59 | Elevated | Mild deviation from historical norms | Monitor positions closely; review exposure |
| 60–74 | High Risk | Significant market anomaly | Review and consider hedging strategies |
| 75–100 | Extreme | Crisis-level market anomaly | Reduce risk exposure; liquidity management |

---

## Deployment

### Development Setup

| Component | Configuration |
|-----------|---|
| **Frontend** | Vite dev server @ `http://localhost:5173` (HMR enabled) |
| **Backend** | Uvicorn @ `http://localhost:8000` (auto-reload, Swagger @ `/docs`) |
| **Models** | In-memory PyTorch (54 total: 9 models × 6 assets) |

### Production Deployment

| Component | Stack |
|-----------|---|
| **Frontend** | AWS S3 + CloudFront CDN |
| **Backend** | Docker → Kubernetes (auto-scaling) |
| **Models** | Redis cache + TorchServe |
| **Database** | PostgreSQL / TimescaleDB |
| **Monitoring** | Prometheus + Grafana |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Overview

A comprehensive production-ready system demonstrating end-to-end ML engineering: from raw market data ingestion and feature engineering, through 9-model anomaly detection and price forecasting, to a scalable REST API and interactive dashboard with **multi-mode anomaly forecasting** (ensemble, advanced, DL composite, and regime-aware hybrid modes). Features real-time comparison overlays, regime-adaptive blending, and institutional-grade risk monitoring for algorithmic trading and quantitative research.
