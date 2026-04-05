import { useState, useCallback } from 'react'
import Layout from './components/layout/Layout'
import { ToastContainer } from './components/ui/Toast'
import Dashboard from './pages/Dashboard'
import Forecast from './pages/Forecast'
import Historical from './pages/Historical'
import Regime from './pages/Regime'
import AdvancedAnomaly from './pages/AdvancedAnomaly'
import { useMarketData } from './hooks/useMarketData'
import { useAutoRefresh } from './hooks/useAutoRefresh'

let _toastId = 0

export default function App() {
  const [selectedAsset, setSelectedAsset] = useState('SP500')
  const [activePage, setActivePage] = useState('dashboard')
  const [toasts, setToasts] = useState([])

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

  const renderPage = () => {
    switch (activePage) {
      case 'forecast':
        return (
          <Forecast
            priceForecast={priceForecast}
            selectedAsset={selectedAsset}
            loading={loading}
            error={error}
          />
        )
      case 'historical':
        return (
          <Historical
            historical={historical}
            selectedAsset={selectedAsset}
            loading={loading}
            error={error}
          />
        )
      case 'regime':
        return (
          <Regime
            asset={selectedAsset}
            loading={loading}
          />
        )
      case 'advanced-anomaly':
        return (
          <AdvancedAnomaly
            selectedAsset={selectedAsset}
            loading={loading}
          />
        )
      default:
        return (
          <Dashboard
            current={current}
            historical={historical}
            forecast={forecast}
            priceForecast={priceForecast}
            evaluation={evaluation}
            comparison={comparison}
            loading={loading}
            error={error}
            selectedAsset={selectedAsset}
          />
        )
    }
  }

  return (
    <>
      <Layout
        activePage={activePage}
        onPageChange={setActivePage}
        selectedAsset={selectedAsset}
        onAssetChange={setSelectedAsset}
        apiOnline={apiOnline}
        loading={loading}
        onRefresh={refresh}
        lastUpdated={lastUpdated}
        sp500Score={sp500Score}
      >
        {renderPage()}
      </Layout>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </>
  )
}
