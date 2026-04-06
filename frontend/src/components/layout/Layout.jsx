import { useState } from 'react'
import { motion } from 'framer-motion'
import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'
import Sidebar from './Sidebar'
import { PERIODS } from '../../constants/config'

export default function Layout({
  selectedAsset, onAssetChange,
  apiOnline, loading, onRefresh, lastUpdated,
  sp500Score,
}) {
  const [period, setPeriod] = useState(PERIODS[3]) // 1Y default
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  return (
    <div className="relative min-h-screen bg-page-bg text-text-primary">
      <Navbar
        selectedAsset={selectedAsset}
        onAssetChange={onAssetChange}
        selectedPeriod={period}
        onPeriodChange={setPeriod}
        apiOnline={apiOnline}
        loading={loading}
        onRefresh={onRefresh}
        lastUpdated={lastUpdated}
      />
      <Sidebar
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        sp500Score={sp500Score}
      />
      <motion.main
        className="relative pt-16 transition-all duration-300 ease-in-out"
        animate={{ paddingLeft: sidebarCollapsed ? '5rem' : '16rem' }}
        layout
      >
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <Outlet />
        </div>
      </motion.main>
    </div>
  )
}
