import { useState, useCallback, useEffect } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import Layout from './components/layout/Layout'
import { ToastContainer } from './components/ui/Toast'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Anomalies from './pages/Anomalies'
import Forecast from './pages/Forecast'
import Historical from './pages/Historical'
import Regime from './pages/Regime'
import Evaluation from './pages/Evaluation'
import { useMarketData } from './hooks/useMarketData'
import { useAutoRefresh } from './hooks/useAutoRefresh'

let _toastId = 0

export default function App() {
  const [selectedAsset, setSelectedAsset] = useState('SP500')
  const [toasts, setToasts] = useState([])
  const location = useLocation()
  
  const getActivePageFromPath = (path) => {
    const segments = path.split('/')
    if (segments.length >= 3 && segments[1] === 'app') {
      return segments[2]
    }
    return 'dashboard'
  }
  
  const activePage = getActivePageFromPath(location.pathname)

  const {
    current, historical, forecast, priceForecast,
    comparison, evaluation, summary,
    apiOnline, loading, error,
    lastUpdated, refresh,
  } = useMarketData(selectedAsset, activePage)

  useAutoRefresh(refresh)

  const addToast = useCallback((message, type = 'info') => {
    const id = ++_toastId
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const sp500Score = summary?.assets?.find?.(s => s?.asset === 'SP500')?.anomaly_score ?? null

  return (
    <>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route
          path="/app"
          element={
            <Layout
              selectedAsset={selectedAsset}
              onAssetChange={setSelectedAsset}
              apiOnline={apiOnline}
              loading={loading}
              onRefresh={refresh}
              lastUpdated={lastUpdated}
              sp500Score={sp500Score}
            />
          }
        >
          <Route index element={<Navigate to="/app/dashboard" replace />} />
          <Route
            path="dashboard"
            element={
              <Dashboard
                current={current}
                historical={historical}
                forecast={forecast}
                priceForecast={priceForecast}
                evaluation={evaluation}
                comparison={comparison}
                summary={summary}
                loading={loading}
                error={error}
                selectedAsset={selectedAsset}
                onSelectAsset={setSelectedAsset}
                apiOnline={apiOnline}
                lastUpdated={lastUpdated}
                onRefresh={refresh}
              />
            }
          />
          <Route
            path="anomalies"
            element={
              <Anomalies
                selectedAsset={selectedAsset}
                loading={loading}
              />
            }
          />
          <Route
            path="forecast"
            element={
              <Forecast
                priceForecast={priceForecast}
                anomalyForecast={forecast}
                currentAnalysis={current}
                selectedAsset={selectedAsset}
                loading={loading}
                error={error}
              />
            }
          />
          <Route
            path="historical"
            element={
              <Historical
                historical={historical}
                selectedAsset={selectedAsset}
                loading={loading}
                error={error}
              />
            }
          />
          <Route
            path="regime"
            element={
              <Regime
                asset={selectedAsset}
                loading={loading}
              />
            }
          />
          <Route
            path="evaluation"
            element={
              <Evaluation
                evaluation={evaluation}
                historical={historical}
                selectedAsset={selectedAsset}
                loading={loading}
              />
            }
          />
        </Route>
      </Routes>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </>
  )
}
