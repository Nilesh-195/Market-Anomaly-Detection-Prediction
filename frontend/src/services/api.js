import axios from 'axios'
import { API_BASE } from '../constants/config'

const client = axios.create({ baseURL: API_BASE, timeout: 15000 })

export async function fetchAssets() {
  const { data } = await client.get('/assets')
  return data
}

export async function fetchCurrentAnalysis(ticker) {
  const { data } = await client.get(`/current-analysis/${ticker}`)
  return data
}

export async function fetchHistoricalAnomalies(ticker, topN = 50) {
  const { data } = await client.get(`/historical-anomalies/${ticker}`, {
    params: { top_n: topN },
  })
  return data
}

export async function fetchForecast(ticker, days = 10) {
  const { data } = await client.get(`/forecast/${ticker}`, {
    params: { days },
  })
  return data
}

export async function fetchModelComparison(ticker) {
  const { data } = await client.get(`/model-comparison/${ticker}`)
  return data
}

export async function fetchEvaluation() {
  const { data } = await client.get('/evaluation')
  return data
}

export async function fetchSummary() {
  const { data } = await client.get('/summary')
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
