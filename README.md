# 📈 Market Anomaly Detection & Prediction

> **A production-grade, end-to-end AI system for stock price forecasting and market anomaly detection across 6 financial assets — powered by a multi-model ensemble, served via a FastAPI backend, and visualised in a live 6-page React dashboard.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite&logoColor=white)](https://vite.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API Version](https://img.shields.io/badge/API-v2.1.0-brightgreen)](http://localhost:8000/docs)

---

## 🌟 Project Highlights

| Metric | Value |
|--------|-------|
| **Assets covered** | 6 — S&P 500, VIX, Bitcoin, Gold, NASDAQ, Tesla |
| **Historical data** | 2010 → 2026 (~4,000 trading days per asset) |
| **Labelled crash events** | 13 major market events (Flash Crash → Yen Carry Unwind) |
| **Anomaly models** | 7 (Baseline: 4 · Advanced Phase 2: 3) × 6 assets |
| **Forecasting methods** | 9 (Naive · Drift · SES · Holt · ARIMA · SARIMA · VAR · LSTM · Transformer · XGBoost) |
| **Ensemble ROC-AUC** | **0.74 average** (SP500: 0.84 · NASDAQ: 0.80 · VIX: 0.83) |
| **Crash hit-rate** | **38% average** (TESLA: 67% · NASDAQ: 54% · VIX: 46%) |
| **Training time** | ~74 seconds total (Apple Silicon MPS) |
| **API endpoints** | 20+ REST endpoints (FastAPI + WebSocket + CORS) |
| **Dashboard** | 6-page live React UI with light/dark theme & interactive charts |

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Features](#-features)
- [Models](#-models)
- [Forecasting Methods](#-forecasting-methods)
- [Engineered Features](#-engineered-features-15-per-asset)
- [Project Structure](#️-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Labelled Crash Events](#-labelled-crash-events-13)
- [Model Evaluation](#-model-evaluation-results)
- [Tech Stack](#-tech-stack)
- [Risk Score Interpretation](#-risk-score-interpretation)

---

## 🏗️ Architecture

### System Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🎨  PRESENTATION LAYER                              ║
║                   React 19 Dashboard  ·  Vite 7  ·  Recharts               ║
║                                                                              ║
║   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   ║
║   │Dashboard │  │ Forecast │  │Historical│  │  Regime  │  │Advanced  │   ║
║   │(Overview)│  │  (Price) │  │(Crashes) │  │ Detection│  │ Anomaly  │   ║
║   └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   ║
║   • 6 asset selector cards        • Light / Dark theme toggle              ║
║   • Live ensemble risk gauge       • Auto-refresh every 5 minutes          ║
║   • Interactive Brush zoom         • Toast notification system             ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                        HTTP / JSON / WebSocket / CORS
                                     │
                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀  API LAYER  ·  FastAPI 0.135  ·  Uvicorn             ║
║                         Port 8000  ·  Auto-Docs /docs                      ║
║                                                                              ║
║   Forecasting Group              │  Anomaly Detection Group                 ║
║   ────────────────────────────── │  ──────────────────────────────          ║
║   /forecast/price/{asset}        │  /anomaly/current/{asset}               ║
║   /forecast/arima/{asset}        │  /anomaly/advanced/{asset}              ║
║   /forecast/naive/{asset}        │  /anomaly/regime/{asset}                ║
║   /forecast/exponential/{asset}  │  /anomaly/historical/{asset}            ║
║   /forecast/var                  │  /anomaly/compare-tiers/{asset}         ║
║   /forecast/compare/{asset}      │  /anomaly/forecast/{asset}              ║
║   /forecast/stationarity/{asset} │  /anomaly/comparison/{asset}            ║
║   /forecast/acf-pacf/{asset}     │                                         ║
║   /forecast/evaluate/{asset}     │  General: /  /assets  /summary          ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠  INFERENCE LAYER  ·  Model Registry                  ║
║                                                                              ║
║   ── FORECASTING MODELS ──────────────────────────────────────────────────  ║
║   Naive / Drift / Mean   │  SES / Holt / Holt-Winters  │  ARIMA / SARIMA   ║
║   VAR (multi-asset)      │  LSTM Seq2Seq (PyTorch)      │  Transformer      ║
║   XGBoost                │                                                  ║
║                                                                              ║
║   ── ANOMALY DETECTION MODELS (per asset × 6) ────────────────────────────  ║
║   Baseline:  Z-Score · Isolation Forest · LSTM Autoencoder · Prophet        ║
║   Advanced:  XGBoost (supervised) · HMM (regime) · TCN (temporal)          ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                 ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                  🗄️  DATA LAYER  ·  Feature Store                          ║
║                                                                              ║
║   yfinance OHLCV (2010 → 2026)  →  15 Engineered Features  →  Parquet      ║
║   Crash Labels JSON (13 events)  →  ±5 day detection window                ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### Ensemble Scoring (Anomaly Detection)

```
Ensemble Score = (15% × Z-Score)
               + (25% × Isolation Forest)
               + (40% × LSTM Autoencoder)
               + (20% × Prophet Residual)

Final Score Range: 0 – 100  (normalized)

  0 – 39   →  🟢 Normal         (no significant anomaly)
 40 – 59   →  🟡 Elevated       (monitor positions closely)
 60 – 74   →  🟠 High Risk      (review exposure & hedge)
 75 – 100  →  🔴 Extreme        (crisis-level alert)
```

---

## ✨ Features

### 📊 Stock Price Forecasting *(Primary)*
- **9 forecasting methods** from naive baselines to deep learning
- **Automatic method selection** via cross-validated RMSE ranking
- **95% prediction intervals** for all methods
- **Stationarity analysis** — ADF & KPSS tests with differencing recommendation
- **ACF/PACF analysis** — ARIMA order (p, d, q) auto-suggestion
- **Multi-asset VAR** — Granger causality between SP500, NASDAQ & VIX
- **Method comparison** — ranked table across all classical & DL methods

### 🔍 Market Anomaly Detection *(Bonus)*
- **4-model baseline ensemble** — Z-Score, Isolation Forest, LSTM Autoencoder, Prophet
- **3-model advanced extension** — XGBoost supervised, HMM regime, TCN temporal
- **Market regime detection** — bull / bear / crisis classification via HMM
- **Historical crash analysis** — 13 labelled events with ±5 day detection window
- **SHAP explainability** — feature importance for advanced anomaly scores
- **Tier comparison** — baseline vs. advanced ensemble overlay

### 🖥️ Live React Dashboard
- **6-page navigation** — Dashboard, Forecast, Historical, Regime, Advanced Anomaly, Settings
- **Light / Dark theme** — fully-designed dual-theme toggle
- **Interactive chart zoom** — Recharts Brush component for all time-series charts
- **Auto-refresh** — market data auto-updates every 5 minutes
- **Toast notifications** — real-time feedback for data loading and errors
- **Asset switcher** — instant context switch across all 6 assets

---

## 🤖 Models

### Anomaly Detection — Baseline (Phase 1)

| # | Model | Library | Key Config |
|---|-------|---------|-----------|
| 1 | **Z-Score** | Pure numpy | 20-day rolling window on log-returns |
| 2 | **Isolation Forest** | scikit-learn | 200 trees · contamination = 3% · 15 features |
| 3 | **LSTM Autoencoder** | PyTorch 2.10 | Encoder 15→64→8 · Decoder 8→64→15 · 30-day window |
| 4 | **Prophet Residual** | Facebook Prophet | Pre-2020 training · anomaly = \|actual − yhat\| / 3σ |

### Anomaly Detection — Advanced (Phase 2)

| # | Model | Library | Key Config |
|---|-------|---------|-----------|
| 5 | **XGBoost Supervised** | xgboost 2.1 | Trained on 13 labelled crash events + 15 features |
| 6 | **HMM Regime** | hmmlearn | 3 hidden states (bull / bear / crisis) |
| 7 | **TCN (Temporal)** | PyTorch | Dilated causal convolutions · 30-day receptive field |

### Forecasting Models (Phase 3)

| Category | Method | Implementation |
|----------|--------|----------------|
| Naive Baselines | Mean, Naïve, Seasonal Naïve, Drift | Custom NumPy |
| Exponential Smoothing | SES, Holt's Linear, Damped Trend, Holt-Winters | `statsmodels` |
| ARIMA Family | ARIMA, SARIMA (auto-order selection) | `statsmodels` |
| Multivariate | VAR (Vector Auto-Regression) + Granger causality | `statsmodels` |
| Deep Learning | LSTM Seq2Seq, Temporal Transformer | PyTorch 2.10 |
| Gradient Boosting | XGBoost with lagged & cyclic features | `xgboost` |

---

## 📊 Engineered Features (15 per asset)

| Feature | Category | Description |
|---------|----------|-------------|
| `log_return` | Returns | Daily log return |
| `zscore_10`, `zscore_20`, `zscore_60` | Statistical | Rolling Z-score (10, 20, 60-day) |
| `vol_10`, `vol_30` | Volatility | Rolling annualised volatility |
| `vol_ratio` | Volatility | Short / long volatility ratio |
| `drawdown` | Risk | Rolling 252-day maximum drawdown |
| `bubble_score` | Trend | Price / 200-day SMA deviation |
| `rsi_14` | Technical | RSI normalised to [0, 1] |
| `bb_position` | Technical | Position within Bollinger Bands |
| `macd_hist` | Technical | MACD histogram (normalised) |
| `volume_zscore` | Volume | Volume anomaly (20-day Z-score) |
| `vwap_deviation` | Volume | Price deviation from VWAP |
| `atr_ratio` | Volatility | ATR / price (normalised) |

---

## 🗂️ Project Structure

```
Market_Anomaly_Detection_Prediction/
├── backend/
│   ├── api/
│   │   └── main.py                  # FastAPI v2.1.0 — 20+ REST endpoints
│   ├── data/
│   │   ├── raw/                     # 6 × parquet (OHLCV, 2010–2026)
│   │   ├── processed/               # 6 × feature parquet (15 features)
│   │   └── crash_labels.json        # 13 labelled market crash events
│   ├── models/                      # Trained model artifacts (gitignored)
│   │   └── <ASSET>/
│   │       ├── isolation_forest.pkl
│   │       ├── lstm_autoencoder.pt
│   │       ├── lstm_meta.pkl
│   │       ├── prophet_model.pkl
│   │       └── scores_all.parquet
│   ├── notebooks/
│   │   └── 01_eda.ipynb             # EDA — 11 cells, 7 charts
│   ├── src/
│   │   ├── data_loader.py           # yfinance download + crash label management
│   │   ├── features.py              # 15-feature engineering pipeline
│   │   ├── models.py                # Baseline anomaly model classes + ensemble
│   │   ├── advanced_models.py       # Phase 2 — XGBoost, HMM, TCN
│   │   ├── train.py                 # Training orchestrator (all 6 assets)
│   │   ├── predict.py               # Inference — current / forecast / history
│   │   ├── evaluate.py              # Precision / Recall / F1 / AUC / hit-rate
│   │   ├── stationarity.py          # ADF & KPSS tests
│   │   ├── acf_pacf_analysis.py     # ACF/PACF + ARIMA order suggestion
│   │   ├── naive_methods.py         # Mean, Naïve, Seasonal Naïve, Drift
│   │   ├── exponential_smoothing.py # SES, Holt, Holt-Winters
│   │   ├── arima_models.py          # ARIMA / SARIMA
│   │   ├── var_model.py             # Multi-asset VAR + Granger causality
│   │   ├── forecast_dl.py           # LSTM Seq2Seq, Transformer, XGBoost (Phase 3)
│   │   ├── forecasting.py           # Unified forecasting orchestrator
│   │   └── forecast_evaluation.py   # Cross-validation + method ranking
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Root — routing + state management
│   │   ├── index.css                # Global design system (light/dark tokens)
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        # Overview — ensemble gauge + sparklines
│   │   │   ├── Forecast.jsx         # Price forecast with method selector
│   │   │   ├── Historical.jsx       # Crash event timeline
│   │   │   ├── Regime.jsx           # HMM market regime visualisation
│   │   │   └── AdvancedAnomaly.jsx  # Phase 2 — 7-model anomaly panel
│   │   ├── components/
│   │   │   ├── layout/              # Sidebar, Navbar, Layout shell
│   │   │   └── ui/                  # Toast, Spinner, Card, Badge, etc.
│   │   ├── hooks/
│   │   │   ├── useMarketData.js     # Unified data-fetching hook
│   │   │   └── useAutoRefresh.js    # 5-minute polling hook
│   │   ├── services/                # Axios API service layer
│   │   ├── constants/               # Asset list, colour tokens
│   │   └── utils/                   # Formatters, helpers
│   ├── package.json
│   └── vite.config.js
├── testing_summary.md
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### 1 — Clone & set up Python environment

```bash
git clone https://github.com/Nilesh-195/Market-Anomaly-Detection-Prediction.git
cd Market-Anomaly-Detection-Prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2 — Download data & build features

```bash
python backend/src/data_loader.py   # downloads 6 assets → data/raw/
python backend/src/features.py      # builds 15 features → data/processed/
```

### 3 — Train anomaly detection models

```bash
python backend/src/train.py         # ~2 min on Apple Silicon / ~15 min on CPU
```

### 4 — Run evaluation

```bash
python backend/src/evaluate.py      # outputs evaluation_report.json + .csv
```

### 5 — Start the API server

```bash
uvicorn backend.api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 6 — Start the React dashboard

```bash
cd frontend
npm install
npm run dev
# Dashboard → http://localhost:5173
```

---

## 🔌 API Reference

### General

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + API info |
| `GET` | `/assets` | List all 6 supported assets |
| `GET` | `/summary` | Current price, forecast & anomaly score for all assets |

### Forecasting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/forecast/price/{asset}?horizon=30&method=auto` | **Primary** — best method price forecast with 95% CI |
| `GET` | `/forecast/naive/{asset}?horizon=30` | All 4 naive baseline forecasts |
| `GET` | `/forecast/exponential/{asset}?horizon=30&method=auto` | SES / Holt / Holt-Winters |
| `GET` | `/forecast/arima/{asset}?horizon=30&p=1&d=1&q=1&seasonal=false` | ARIMA / SARIMA |
| `GET` | `/forecast/var?assets=SP500,NASDAQ,VIX&horizon=30` | Multi-asset VAR forecast |
| `GET` | `/forecast/compare/{asset}?horizon=30` | Ranked comparison of all methods |
| `GET` | `/forecast/stationarity/{asset}` | ADF & KPSS stationarity tests |
| `GET` | `/forecast/acf-pacf/{asset}?max_lags=40` | ACF/PACF + ARIMA order suggestion |
| `GET` | `/forecast/evaluate/{asset}` | Cross-validated RMSE/MAE/MAPE for all methods |

### Anomaly Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/anomaly/current/{asset}` | Latest baseline ensemble score + model breakdown |
| `GET` | `/anomaly/advanced/{asset}` | All 7 models (baseline + Phase 2) + advanced ensemble |
| `GET` | `/anomaly/regime/{asset}` | HMM market regime timeline (bull/bear/crisis) |
| `GET` | `/anomaly/historical/{asset}` | Top anomaly events, clustered & de-duplicated |
| `GET` | `/anomaly/forecast/{asset}?days=10` | ARIMA 5–30 day anomaly score forecast with CI |
| `GET` | `/anomaly/comparison/{asset}` | Per-model stats + correlation with ensemble |
| `GET` | `/anomaly/compare-tiers/{asset}` | Baseline vs. Advanced ensemble comparison |

### Example Request

```bash
# Get best 30-day price forecast for S&P 500
curl http://localhost:8000/forecast/price/SP500?horizon=30
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
# Get current anomaly score for Bitcoin
curl http://localhost:8000/anomaly/current/BTC
```

```json
{
  "asset": "BTC",
  "date": "2026-03-24",
  "ensemble_score": 41.5,
  "risk_label": "Elevated",
  "model_scores": {
    "zscore": 18.2,
    "iforest": 35.6,
    "lstm": 52.3,
    "prophet": 38.9
  }
}
```

---

## 📉 Labelled Crash Events (13)

| Date | Event | Impact Scale |
|------|-------|-------------|
| 2010-05-06 | **Flash Crash** | 🔴 High |
| 2011-08-08 | **US Debt Downgrade** | 🔴 High |
| 2015-08-24 | **China Black Monday** | 🔴 High |
| 2018-02-05 | **Volmageddon** | 🔴 High |
| 2018-12-24 | **Christmas Eve Crash** | 🟠 Medium |
| 2020-02-24 | **COVID-19 First Wave** | ⚫ Extreme |
| 2020-03-16 | **COVID-19 Peak Crash** | ⚫ Extreme |
| 2021-01-28 | **GameStop Short Squeeze** | 🟠 Medium |
| 2022-01-24 | **Fed Tightening Panic** | 🔴 High |
| 2022-05-12 | **Luna / Terra Collapse** | ⚫ Extreme |
| 2022-09-28 | **UK Gilt Crisis** | 🟠 Medium |
| 2023-03-10 | **SVB Bank Collapse** | 🔴 High |
| 2024-08-05 | **Yen Carry Trade Unwind** | 🔴 High |

---

## 📊 Model Evaluation Results

Evaluated against the 13 crash events using a ±5 trading-day detection window:

| Asset | Ensemble AUC | Hit Rate | Events Detected |
|-------|:-----------:|:--------:|:--------------:|
| S&P 500 | **0.841** | 23% | 3 / 13 |
| VIX | **0.832** | 46% | 6 / 13 |
| Bitcoin | 0.633 | 36% | 4 / 11 |
| Gold | 0.607 | 0% | 0 / 13 |
| NASDAQ | **0.803** | 54% | 7 / 13 |
| Tesla | 0.705 | 67% | 8 / 12 |
| **Mean** | **0.737** | **38%** | — |

> **Note:** Low hit-rates are expected for unsupervised models trained pre-2020. The strong AUC scores (0.74 avg) confirm the ensemble correctly *ranks* anomalous days above normal ones without any labelled supervision. GOLD's 0% hit-rate reflects that gold acts as a safe-haven and typically moves inversely to crash events.

---

## 🧰 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Data Ingestion** | `yfinance`, `pandas`, `pyarrow` | Latest |
| **Feature Engineering** | `ta` (technical analysis), `numpy`, `scipy` | Latest |
| **Anomaly ML** | `scikit-learn` (Isolation Forest), `PyTorch` (LSTM) | 1.8 / 2.10 |
| **Advanced Anomaly** | `xgboost` (supervised), `hmmlearn` (regime), `shap` | 2.1 / 0.3 / 0.46 |
| **Forecasting** | `statsmodels` (ARIMA/VAR), `PyTorch` (LSTM/Transformer) | 0.14 / 2.10 |
| **Prophet** | `prophet` (Facebook / Meta) | 1.3 |
| **Backend** | `FastAPI`, `uvicorn`, `pydantic` | 0.135 / 0.41 / 2.12 |
| **Frontend** | `React 19`, `Vite 7` | 19 / 7 |
| **Charts** | `Recharts` (with Brush zoom) | Latest |
| **Icons** | `lucide-react` | Latest |
| **Macroeconomic Data** | `fredapi`, `python-dotenv` | 0.5 / 1.0 |

---

## 🔮 Risk Score Interpretation

| Score | Label | Recommended Action |
|------:|-------|-------------------|
| 0 – 39 | 🟢 **Normal** | No significant anomaly detected — hold positions |
| 40 – 59 | 🟡 **Elevated** | Mild deviation from historical norms — monitor closely |
| 60 – 74 | 🟠 **High Risk** | Significant anomaly detected — review exposure & consider hedging |
| 75 – 100 | 🔴 **Extreme** | Major market stress — crisis-level signal, reduce risk |

---

## 🔧 Environment & Deployment

| Environment | Frontend | Backend | Models |
|---|---|---|---|
| **Development** | Vite dev server (localhost:5173) | Uvicorn (localhost:8000) | In-memory cache |
| **Production** | AWS S3 + CloudFront | Docker + K8s | Redis + TorchServe |

```
Current local setup:
  ✅ Frontend  →  npm run dev  (Vite HMR at :5173)
  ✅ Backend   →  uvicorn backend.api.main:app --reload --port 8000
  ✅ Models    →  All 24 baseline + Phase 2 models loaded in-memory
  ✅ Swagger   →  http://localhost:8000/docs
  ✅ Dashboard →  http://localhost:5173
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built as a comprehensive, end-to-end ML engineering project — from raw OHLCV ingestion and feature engineering, through multi-model training and ensemble scoring, to a production-ready REST API and a live interactive dashboard.*
