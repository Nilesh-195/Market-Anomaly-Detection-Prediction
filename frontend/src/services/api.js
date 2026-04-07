import axios from 'axios'
import { API_BASE } from '../constants/config'

const client = axios.create({ baseURL: API_BASE, timeout: 30000 })

export async function fetchAssets() {
  const { data } = await client.get('/assets')
  return data
}

export async function fetchCurrentAnalysis(ticker) {
  const { data } = await client.get(`/anomaly/current/${ticker}`)
  return data
}

export async function fetchHistoricalAnomalies(ticker, topN = 50, threshold = 60) {
  const { data } = await client.get(`/anomaly/historical/${ticker}`, {
    params: { top_n: topN, threshold },
  })
  return data
}

export async function fetchAnomalyForecast(ticker, days = 10) {
  const { data } = await client.get(`/anomaly/forecast/${ticker}`, {
    params: { days },
  })
  return data
}

export async function fetchModelComparison(ticker) {
  const { data } = await client.get(`/anomaly/comparison/${ticker}`)
  return data
}

export async function fetchEvaluation() {
  const { data } = await client.get('/anomaly/evaluation')
  return data
}

export async function fetchSummary() {
  const { data } = await client.get('/summary')
  return data
}

// ═══════════════════════════════════════════════════════════════════════════
// PRICE FORECASTING ENDPOINTS (PRIMARY)
// ═══════════════════════════════════════════════════════════════════════════

export async function fetchPriceForecast(ticker, horizon = 30, method = 'auto') {
  const { data } = await client.get(`/forecast/price/${ticker}`, {
    params: { horizon, method },
  })
  return data
}

export async function fetchForecastComparison(ticker, horizon = 30) {
  const { data } = await client.get(`/forecast/compare/${ticker}`, {
    params: { horizon },
  })
  return data
}

export async function fetchAcfPacf(ticker, maxLags = 40) {
  const { data } = await client.get(`/forecast/acf-pacf/${ticker}`, {
    params: { max_lags: maxLags },
  })
  return data
}

export async function fetchStationarity(ticker) {
  const { data } = await client.get(`/forecast/stationarity/${ticker}`)
  return data
}

export async function fetchNaiveForecast(ticker, horizon = 30) {
  const { data } = await client.get(`/forecast/naive/${ticker}`, {
    params: { horizon },
  })
  return data
}

export async function fetchExponentialForecast(ticker, horizon = 30, method = 'auto') {
  const { data } = await client.get(`/forecast/exponential/${ticker}`, {
    params: { horizon, method },
  })
  return data
}

export async function fetchArimaForecast(ticker, horizon = 30, seasonal = false) {
  const { data } = await client.get(`/forecast/arima/${ticker}`, {
    params: { horizon, seasonal },
  })
  return data
}

export async function checkHealth() {
  try {
    const { data } = await client.get('/', { timeout: 3000 })
    return data?.status === 'ok'
  } catch {
    return false
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// DEEP LEARNING FORECASTING (Phase 3 - NEW)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Normalize the flat DL/XGBoost response format into the standard nested
 * format that Forecast.jsx expects.
 * Flat:   { forecast: [...], lower_95: [...], upper_95: [...], dates: [...], ... }
 * Nested: { current_price, horizon, method, forecast: { values, lower_95, upper_95, dates }, summary }
 */
function _normalizeDlResponse(data) {
  if (!data) return data
  // Already in nested format (has forecast.values)
  if (data.forecast && !Array.isArray(data.forecast)) return data

  const forecastArr = Array.isArray(data.forecast) ? data.forecast : []
  const currentPrice = forecastArr[0] ?? 0   // best approximation when unavailable
  const lastPrice = forecastArr[forecastArr.length - 1] ?? currentPrice
  const expectedReturn = currentPrice > 0
    ? parseFloat((((lastPrice / currentPrice) - 1) * 100).toFixed(2))
    : 0

  return {
    asset: data.asset,
    current_price: data.current_price ?? currentPrice,
    horizon: data.horizon ?? forecastArr.length,
    method: data.method,
    model_info: data.model_info ?? {},
    forecast: {
      values: forecastArr,
      lower_95: data.lower_95 ?? forecastArr,
      upper_95: data.upper_95 ?? forecastArr,
      dates: data.dates ?? [],
    },
    summary: {
      forecast_30d: lastPrice,
      expected_return_pct: expectedReturn,
    },
    // Pass through extras for charts (attention, feature importance)
    attention_weights: data.attention_weights ?? null,
    feature_importance: data.feature_importance ?? null,
  }
}

export async function fetchLstmForecast(ticker, horizon = 30) {
  const { data } = await client.get(`/forecast/lstm/${ticker}`, {
    params: { horizon },
  })
  return _normalizeDlResponse(data)
}

export async function fetchTransformerForecast(ticker, horizon = 30) {
  const { data } = await client.get(`/forecast/transformer/${ticker}`, {
    params: { horizon },
  })
  return _normalizeDlResponse(data)
}

export async function fetchXgboostForecast(ticker, horizon = 30) {
  const { data } = await client.get(`/forecast/xgboost-price/${ticker}`, {
    params: { horizon },
  })
  return _normalizeDlResponse(data)
}

// ═══════════════════════════════════════════════════════════════════════════
// ADVANCED ANOMALY DETECTION (Phase 2 - NEW)
// ═══════════════════════════════════════════════════════════════════════════

export async function fetchAdvancedAnomaly(ticker) {
  const { data } = await client.get(`/anomaly/advanced/${ticker}`)
  return data
}

export async function fetchRegimeTimeline(ticker) {
  const { data } = await client.get(`/anomaly/regime/${ticker}`)
  return data
}

export async function fetchCompareTiers(ticker) {
  const { data } = await client.get(`/anomaly/compare-tiers/${ticker}`)
  return data
}

export async function fetchCrashEvents(asset, fromDate, toDate) {
  const params = {}
  if (asset) params.asset = asset
  if (fromDate) params.from_date = fromDate
  if (toDate) params.to_date = toDate

  const { data } = await client.get('/events/crashes', { params })
  return data
}

export async function fetchCrashLabels() {
  const { data } = await client.get('/anomaly/crash-labels')
  return data
}

export async function fetchAnomalyMetrics(ticker, threshold = 60, model = 'ensemble_score', windowDays = 7) {
  const { data } = await client.get(`/anomaly/metrics/${ticker}`, {
    params: {
      threshold,
      model,
      window_days: windowDays,
    },
  })
  return data
}

export async function fetchThresholdAnalysis(
  ticker,
  {
    model = 'ensemble_score',
    minThreshold = 40,
    maxThreshold = 80,
    step = 2,
    costFp = 1,
    costFn = 5,
    windowDays = 7,
  } = {}
) {
  const { data } = await client.get(`/anomaly/threshold-analysis/${ticker}`, {
    params: {
      model,
      min_threshold: minThreshold,
      max_threshold: maxThreshold,
      step,
      cost_fp: costFp,
      cost_fn: costFn,
      window_days: windowDays,
    },
  })
  return data
}

export async function fetchFalsePositives(ticker, threshold = 60, topN = 40, model = 'ensemble_score', windowDays = 7) {
  const { data } = await client.get(`/anomaly/false-positives/${ticker}`, {
    params: {
      threshold,
      top_n: topN,
      model,
      window_days: windowDays,
    },
  })
  return data
}

export async function fetchBubbleRisk(ticker) {
  const { data } = await client.get(`/anomaly/bubble-risk/${ticker}`)
  return data
}
