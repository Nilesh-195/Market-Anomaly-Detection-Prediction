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

export async function fetchHistoricalAnomalies(ticker, topN = 50) {
  const { data } = await client.get(`/anomaly/historical/${ticker}`, {
    params: { top_n: topN },
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
