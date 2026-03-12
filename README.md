# 📈 Market Anomaly Detection & Prediction

> **An end-to-end AI system that detects and predicts anomalous behaviour across 6 financial assets using a 4-model ensemble (Z-Score · Isolation Forest · LSTM Autoencoder · Prophet), served via a FastAPI backend and a live React dashboard.**

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite)](https://vite.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Project Highlights

| Metric | Value |
|--------|-------|
| Assets covered | 6 (S&P 500, VIX, Bitcoin, Gold, NASDAQ, Tesla) |
| Historical data | 2010 → 2026 (~4,000 trading days per asset) |
| Labelled crash events | 13 (Flash Crash, COVID-19, SVB Collapse, Yen Carry Unwind…) |
| Models trained | 4 per asset = **24 models** total |
| Ensemble ROC-AUC | **0.74 average** (SP500: 0.84 · NASDAQ: 0.80 · VIX: 0.83) |
| Crash hit-rate | **38% average** (TESLA: 67% · NASDAQ: 54% · VIX: 46%) |
| Training time | **~74 seconds** total (Apple Silicon MPS) |
| API endpoints | 7 REST endpoints (FastAPI + CORS) |
| Dashboard | 4-tab live React UI (dark theme, Recharts) |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    React Dashboard (Vite)                    │
│  Asset Cards · Risk Gauge · Forecast Chart · History Table  │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTP / JSON
┌────────────────────────▼─────────────────────────────────────┐
│              FastAPI Backend  :8000                          │
│  /summary · /current-analysis · /forecast · /historical      │
│  /model-comparison · /evaluation · /assets                   │
└──────┬──────────────┬──────────────┬────────────────┬────────┘
       │              │              │                │
  ┌────▼────┐   ┌─────▼─────┐  ┌───▼────┐   ┌──────▼──────┐
  │ Z-Score │   │ Isolation │  │  LSTM  │   │   Prophet   │
  │Baseline │   │  Forest   │  │  Auto- │   │  Residual   │
  │(stat.)  │   │ (sklearn) │  │encoder │   │ (Facebook)  │
  └────┬────┘   └─────┬─────┘  │(PyTorch│   └──────┬──────┘
       │              │        └───┬────┘           │
       └──────────────┴────────────┴────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Ensemble Score  │
                    │   0 – 100       │
                    │ Normal/Elevated │
                    │ High/Extreme    │
                    └─────────────────┘
```

**Weights:** Z-Score 15% · Isolation Forest 25% · LSTM 40% · Prophet 20%

---

## 🤖 Models

### Model 1 — Z-Score Baseline
- Uses 20-day rolling Z-score of log-returns
- No training required — purely statistical
- Captures sudden return shocks instantly

### Model 2 — Isolation Forest
- `sklearn.ensemble.IsolationForest` (200 trees, contamination=3%)
- Trained on 15 engineered features per asset
- Unsupervised outlier detection in high-dimensional feature space

### Model 3 — LSTM Autoencoder *(PyTorch)*
- Encoder: `LSTM(15→64)` → `Linear(64→8)` (bottleneck)
- Decoder: `Linear(8→64)` → `LSTM(64→64)` → `Linear(64→15)`
- 30-day sliding windows; anomaly = reconstruction MSE > threshold
- Trained on Apple Silicon MPS (GPU); early stopping (patience=8)
- **Replaces TensorFlow** — fully compatible with Python 3.14

### Model 4 — Prophet Residual
- Facebook Prophet fitted to closing prices (pre-2020 training)
- Anomaly score = |actual − yhat| / (3 × residual_std)
- Detects structural breaks and trend deviations

---

## 📊 Engineered Features (15 per asset)

| Feature | Description |
|---------|-------------|
| `log_return` | Daily log return |
| `zscore_10/20/60` | Rolling Z-score (10, 20, 60-day) |
| `vol_10`, `vol_30` | Rolling volatility (annualised) |
| `vol_ratio` | Short/long volatility ratio |
| `drawdown` | Rolling 252-day max drawdown |
| `bubble_score` | Price / 200-day SMA deviation |
| `rsi_14` | RSI normalised to [0, 1] |
| `bb_position` | Bollinger Band position |
| `macd_hist` | MACD histogram (normalised) |
| `volume_zscore` | Volume anomaly (20-day Z-score) |
| `vwap_deviation` | Price deviation from VWAP |
| `atr_ratio` | ATR / price (normalised volatility) |

---

## 🗂️ Project Structure

```
Market_Anomaly_Detection_Prediction/
├── backend/
│   ├── api/
│   │   └── main.py            # FastAPI — 7 REST endpoints
│   ├── data/
│   │   ├── raw/               # 6 × parquet (OHLCV, 2010-2026)
│   │   ├── processed/         # 6 × feature parquet (15 features)
│   │   └── crash_labels.json  # 13 labelled market crash events
│   ├── models/                # Trained model artifacts (gitignored)
│   │   └── <ASSET>/
│   │       ├── isolation_forest.pkl
│   │       ├── lstm_autoencoder.pt
│   │       ├── lstm_meta.pkl
│   │       ├── prophet_model.pkl
│   │       └── scores_all.parquet
│   ├── notebooks/
│   │   └── 01_eda.ipynb       # EDA — 11 cells, 7 charts
│   ├── src/
│   │   ├── data_loader.py     # yfinance download + crash labels
│   │   ├── features.py        # 15-feature engineering pipeline
│   │   ├── models.py          # All 4 model classes + ensemble
│   │   ├── train.py           # Training orchestrator
│   │   ├── predict.py         # current / forecast / history / comparison
│   │   └── evaluate.py        # Precision / Recall / F1 / AUC / hit-rate
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main dashboard (4 tabs)
│   │   └── index.css          # Tailwind dark theme
│   ├── package.json
│   └── vite.config.js
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11–3.14
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

### 3 — Train all models

```bash
python backend/src/train.py         # ~2 min on Apple Silicon / ~15 min on CPU
```

### 4 — Run evaluation

```bash
python backend/src/evaluate.py      # outputs evaluation_report.json + .csv
```

### 5 — Start the API

```bash
uvicorn backend.api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 6 — Start the dashboard

```bash
cd frontend
npm install
npm run dev
# Dashboard → http://localhost:5173
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/assets` | List all supported assets |
| `GET` | `/summary` | Current scores for all 6 assets |
| `GET` | `/current-analysis/{asset}` | Latest ensemble score + model breakdown |
| `GET` | `/forecast/{asset}?days=10` | ARIMA 5–30 day score forecast with CI |
| `GET` | `/historical-anomalies/{asset}` | Top anomaly events (clustered, de-duped) |
| `GET` | `/model-comparison/{asset}` | Per-model stats + correlation with ensemble |
| `GET` | `/evaluation` | Full Precision/Recall/F1/AUC report |

**Example:**
```bash
curl http://localhost:8000/current-analysis/SP500
```
```json
{
  "asset": "SP500",
  "date": "2026-03-11",
  "ensemble_score": 33.03,
  "risk_label": "Normal",
  "model_scores": {
    "zscore": 0.85,
    "iforest": 16.27,
    "lstm": 22.09,
    "prophet": 100.0
  }
}
```

---

## 📉 Labelled Crash Events (13)

| Date | Event | Impact |
|------|-------|--------|
| 2010-05-06 | Flash Crash | High |
| 2011-08-08 | US Debt Downgrade | High |
| 2015-08-24 | China Black Monday | High |
| 2018-02-05 | Volmageddon | High |
| 2018-12-24 | Christmas Crash | Medium |
| 2020-02-24 | COVID First Wave | Extreme |
| 2020-03-16 | COVID Peak Crash | Extreme |
| 2021-01-28 | GameStop Short Squeeze | Medium |
| 2022-01-24 | Fed Tightening Panic | High |
| 2022-05-12 | Luna/Terra Collapse | Extreme |
| 2022-09-28 | UK Gilt Crisis | Medium |
| 2023-03-10 | SVB Bank Collapse | High |
| 2024-08-05 | Yen Carry Trade Unwind | High |

---

## 📊 Model Evaluation Results

Evaluated against the 13 crash events using a ±5-trading-day detection window:

| Asset | Ensemble AUC | Hit Rate | Detected |
|-------|-------------|----------|----------|
| SP500 | **0.841** | 23% | 3 / 13 |
| VIX | **0.832** | 46% | 6 / 13 |
| BTC | 0.633 | 36% | 4 / 11 |
| GOLD | 0.607 | 0% | 0 / 13 |
| NASDAQ | **0.803** | 54% | 7 / 13 |
| TESLA | 0.705 | 67% | 8 / 12 |
| **Mean** | **0.737** | **38%** | — |

> **Note:** Low F1 scores are expected for unsupervised anomaly detection trained on pre-2020 data — the high AUC scores (0.74 avg) confirm the models genuinely rank anomalous days above normal days, even without supervision.

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | `yfinance`, `pandas`, `pyarrow` |
| Features | `ta` (technical analysis), `numpy`, `scipy` |
| ML | `scikit-learn` (Isolation Forest), `PyTorch 2.10` (LSTM) |
| Forecasting | `statsmodels` (ARIMA), `prophet` (Facebook Prophet) |
| Backend | `FastAPI`, `uvicorn`, `pydantic` |
| Frontend | `React 19`, `Vite 7`, `Tailwind CSS v3` |
| Charts | `Recharts` |
| Icons | `lucide-react` |

---

## 🔮 Risk Score Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| 0–39 | 🟢 Normal | No significant anomaly detected |
| 40–59 | 🟡 Elevated | Mild deviation — monitor closely |
| 60–74 | 🟠 High Risk | Significant anomaly — review positions |
| 75–100 | 🔴 Extreme Anomaly | Major market stress — crisis-level signal |

---

## 👤 Author

**Nilesh Dwivedi**  
GitHub: [@Nilesh-195](https://github.com/Nilesh-195)

---

*Built as part of a 20-phase end-to-end ML project — from raw data download to live interactive dashboard.*
