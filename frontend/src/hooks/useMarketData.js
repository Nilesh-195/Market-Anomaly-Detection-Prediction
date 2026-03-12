import { useState, useEffect, useCallback } from 'react'
import {
  fetchCurrentAnalysis,
  fetchHistoricalAnomalies,
  fetchForecast,
  fetchModelComparison,
  fetchEvaluation,
  fetchSummary,
  checkHealth,
} from '../services/api'

export function useMarketData(ticker) {
  const [current,    setCurrent]    = useState(null)
  const [historical, setHistorical] = useState(null)
  const [forecast,   setForecast]   = useState(null)
  const [comparison, setComparison] = useState(null)
  const [evaluation, setEvaluation] = useState(null)
  const [summary,    setSummary]    = useState(null)
  const [apiOnline,  setApiOnline]  = useState(null)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const load = useCallback(async () => {
    if (!ticker) return
    setLoading(true)
    setError(null)
    try {
      const online = await checkHealth()
      setApiOnline(online)
      if (!online) throw new Error('API offline')

      const [cur, hist, fore, comp, evl, sum] = await Promise.allSettled([
        fetchCurrentAnalysis(ticker),
        fetchHistoricalAnomalies(ticker, 50),
        fetchForecast(ticker, 10),
        fetchModelComparison(ticker),
        fetchEvaluation(),
        fetchSummary(),
      ])

      if (cur.status      === 'fulfilled') setCurrent(cur.value)
      if (hist.status     === 'fulfilled') setHistorical(hist.value)
      if (fore.status     === 'fulfilled') setForecast(fore.value)
      if (comp.status     === 'fulfilled') setComparison(comp.value)
      if (evl.status      === 'fulfilled') setEvaluation(evl.value)
      if (sum.status      === 'fulfilled') setSummary(sum.value)

      setLastUpdated(new Date())
    } catch (err) {
      setError(err?.message || 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [ticker])

  useEffect(() => { load() }, [load])

  return {
    current, historical, forecast, comparison, evaluation, summary,
    apiOnline, loading, error, lastUpdated, refresh: load,
  }
}
