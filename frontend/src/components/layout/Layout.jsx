import { useState } from 'react'
import { motion } from 'framer-motion'
import Navbar from './Navbar'
import Sidebar from './Sidebar'
import { PERIODS } from '../../constants/config'

export default function Layout({
  children, activePage, onPageChange,
  selectedAsset, onAssetChange,
  apiOnline, loading, onRefresh, lastUpdated,
  sp500Score,
}) {
  const [period, setPeriod] = useState(PERIODS[3]) // 1Y default

  return (
    <div className="min-h-screen bg-page-bg text-text-primary">
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
        activePage={activePage}
        onPageChange={onPageChange}
        sp500Score={sp500Score}
      />
      <motion.main
        className="pt-16 pl-[220px] transition-all duration-200"
        layout
      >
        <div className="max-w-[1400px] mx-auto p-6">
          {children}
        </div>
      </motion.main>
    </div>
  )
}
