# Market Anomaly Detection & Prediction

This project is designed to detect and predict market anomalies using machine learning techniques.

## Project Structure

```
market-anomaly-detection/
├── backend/
│   ├── data/
│   │   ├── raw/              # Raw market data
│   │   └── processed/        # Processed/cleaned data
│   ├── models/               # Saved trained models
│   ├── notebooks/            # Jupyter notebooks for analysis
│   ├── src/                  # Source code modules
│   │   ├── data_loader.py    # Data loading utilities
│   │   ├── features.py       # Feature engineering
│   │   ├── models.py         # Model definitions
│   │   ├── predict.py        # Prediction logic
│   │   └── evaluate.py       # Model evaluation metrics
│   ├── api/
│   │   └── main.py           # FastAPI/Flask backend API
│   ├── train.py              # Model training script
│   └── requirements.txt       # Python dependencies
├── frontend/
│   └── src/
│       ├── components/       # React components
│       ├── pages/            # Page components
│       └── services/
│           └── api.js        # Frontend API client
└── README.md
```

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API: `python api/main.py`

### Frontend Setup
1. Navigate to the frontend directory
2. Install dependencies: `npm install`
3. Start the development server: `npm start`

## Development

- Place raw market data in `backend/data/raw/`
- Use notebooks in `backend/notebooks/` for data exploration
- Train models using `backend/train.py`
- Access predictions through the API endpoints

