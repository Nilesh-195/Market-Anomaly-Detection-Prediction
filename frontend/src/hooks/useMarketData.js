import { useState, useEffect, useCallback } from 'react'
import {
  fetchCurrentAnalysis,
  fetchHistoricalAnomalies,
  fetchAnomalyForecast,
  fetchModelComparison,
  fetchEvaluation,
  fetchSummary,
  fetchPriceForecast,
  checkHealth,
} from '../services/api'

export function useMarketData(ticker, activePage = 'dashboard') {
  const [current,       setCurrent]       = useState(null)
  const [historical,    setHistorical]    = useState(null)
  const [forecast,      setForecast]      = useState(null)
  const [priceForecast, setPriceForecast] = useState(null)
  const [comparison,    setComparison]    = useState(null)
  const [evaluation,    setEvaluation]    = useState(null)
  const [summary,       setSummary]       = useState(null)
  const [apiOnline,     setApiOnline]     = useState(null)
  const [loading,       setLoading]       = useState(true)
  const [error,         setError]         = useState(null)
  const [lastUpdated,   setLastUpdated]   = useState(null)

  const load = useCallback(async () => {
    if (!ticker) return
    setLoading(true)
    setError(null)

    const setFromSettled = (result, setter) => {
      if (result?.status === 'fulfilled') setter(result.value)
    }

    try {
      // Run health check in parallel so it does not block initial UI data.
      const healthPromise = checkHealth()

      const criticalTasks = [
        fetchCurrentAnalysis(ticker),
        fetchSummary(),
      ]

      if (activePage === 'dashboard' || activePage === 'forecast') {
        criticalTasks.push(fetchPriceForecast(ticker, 30, 'auto'))
      }

      if (activePage === 'historical' || activePage === 'evaluation') {
        criticalTasks.push(fetchHistoricalAnomalies(ticker, activePage === 'evaluation' ? 250 : 50))
      }
      
      if (activePage === 'evaluation') {
         criticalTasks.push(fetchEvaluation())
      }

      const critical = await Promise.allSettled(criticalTasks)
      setFromSettled(critical[0], setCurrent)
      setFromSettled(critical[1], setSummary)

      if (activePage === 'dashboard' || activePage === 'forecast') {
        setFromSettled(critical[2], setPriceForecast)
      }

      if (activePage === 'historical' || activePage === 'evaluation') {
        setFromSettled(critical[2], setHistorical)
      }
      
      if (activePage === 'evaluation') {
        setFromSettled(critical[3], setEvaluation)
      }

      // Unblock UI as soon as critical data is available.
      setLoading(false)

      // Fetch secondary dashboard data lazily.
      if (activePage === 'dashboard') {
        const [hist, fore, comp, evl] = await Promise.allSettled([
          fetchHistoricalAnomalies(ticker, 50),
          fetchAnomalyForecast(ticker, 10),
          fetchModelComparison(ticker),
          fetchEvaluation(),
        ])
        setFromSettled(hist, setHistorical)
        setFromSettled(fore, setForecast)
        setFromSettled(comp, setComparison)
        setFromSettled(evl, setEvaluation)
      }

      const online = await healthPromise
      setApiOnline(online)
      if (!online) throw new Error('API offline')

      setLastUpdated(new Date())
    } catch (err) {
      setError(err?.message || 'Failed to load data')
    } finally {
      setLoading(false)
    }
  }, [ticker, activePage])

  useEffect(() => { load() }, [load])

  return {
    current, historical, forecast, priceForecast, comparison, evaluation, summary,
    apiOnline, loading, error, lastUpdated, refresh: load,
  }
}
