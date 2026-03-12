export const API_BASE = 'http://localhost:8000'

export const ASSETS = [
  { ticker: 'SP500',  name: 'S&P 500',  symbol: '^GSPC' },
  { ticker: 'VIX',    name: 'VIX',      symbol: '^VIX'  },
  { ticker: 'BTC',    name: 'Bitcoin',  symbol: 'BTC-USD' },
  { ticker: 'GOLD',   name: 'Gold',     symbol: 'GLD'   },
  { ticker: 'NASDAQ', name: 'Nasdaq',   symbol: 'QQQ'   },
  { ticker: 'TESLA',  name: 'Tesla',    symbol: 'TSLA'  },
]

export const PERIODS = [
  { label: '1M', days: 30  },
  { label: '3M', days: 90  },
  { label: '6M', days: 180 },
  { label: '1Y', days: 252 },
  { label: '2Y', days: 504 },
]

export const REFRESH_INTERVAL_MS = 5 * 60 * 1000   // 5 minutes

export const RISK_THRESHOLDS = {
  normal:   40,
  elevated: 60,
  high:     75,
}
