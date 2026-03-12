import { useState, useCallback } from 'react'
import Layout from './components/layout/Layout'
import { ToastContainer } from './components/ui/Toast'
import Dashboard from './pages/Dashboard'
import Historical from './pages/Historical'
import Forecast from './pages/Forecast'
import ModelStats from './pages/ModelStats'
import { useMarketData } from './hooks/useMarketData'
import { useAutoRefresh } from './hooks/useAutoRefresh'

const PAGES = {
  dashboard: Dashboard,
  historical: Historical,
  forecast: Forecast,
  models: ModelStats,
}

let _toastId = 0

export default function App() {
  const [selectedAsset, setSelectedAsset] = useState('SP500')
  const [activePage, setActivePage]       = useState('dashboard')
  const [toasts, setToasts]               = useState([])

  const {
    current, historical, forecast,
    comparison, evaluation, summary,
    apiOnline, loading, error,
    lastUpdated, refresh,
  } = useMarketData(selectedAsset)

  useAutoRefresh(refresh)

  const addToast = useCallback((message, type = 'info') => {
    const id = ++_toastId
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 5000)
  }, [])

  const removeToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  const sp500Score = summary?.find?.(s => s?.asset === 'SP500')?.ensemble_score ?? null

  const PageComponent = PAGES[activePage] ?? Dashboard

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
        <PageComponent
          current={current}
          historical={historical}
          forecast={forecast}
          evaluation={evaluation}
          comparison={comparison}
          loading={loading}
          error={error}
        />
      </Layout>

      <ToastContainer toasts={toasts} removeToast={removeToast} />
    </>
  )
}

