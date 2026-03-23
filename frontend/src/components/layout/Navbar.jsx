import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { RefreshCw, ChevronDown } from 'lucide-react'
import clsx from 'clsx'
import { ASSETS, PERIODS } from '../../constants/config'
import { formatTime, formatDate } from '../../utils/formatters'

export default function Navbar({
  selectedAsset, onAssetChange,
  selectedPeriod, onPeriodChange,
  apiOnline, loading, onRefresh, lastUpdated,
}) {
  const [time, setTime] = useState(new Date())
  const [assetOpen, setAssetOpen] = useState(false)

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  function isMarketOpen() {
    const now = new Date()
    const day = now.getDay()
    if (day === 0 || day === 6) return false
    const est = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }))
    const h = est.getHours(), m = est.getMinutes()
    return (h > 9 || (h === 9 && m >= 30)) && h < 16
  }

  const currentAsset = ASSETS.find(a => a.ticker === selectedAsset) || ASSETS[0]

  return (
    <header className="fixed top-0 left-0 right-0 z-40 h-16 bg-white border-b border-card-border flex items-center px-6 gap-6 shadow-sm">
      {/* Brand */}
      <div className="flex items-center gap-3 min-w-[200px]">
        <span className="relative flex h-2 w-2">
          <span className={clsx(
            'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
            apiOnline ? 'bg-risk-normal' : 'bg-risk-extreme'
          )} />
          <span className={clsx(
            'relative inline-flex rounded-full h-2 w-2',
            apiOnline ? 'bg-risk-normal' : 'bg-risk-extreme'
          )} />
        </span>
        <div>
          <div className="font-mono font-bold text-text-primary text-sm tracking-wider">MARKET ANOMALY</div>
          <div className="text-text-secondary text-[10px] tracking-wide">Detection & Forecasting</div>
        </div>
      </div>

      {/* Center controls */}
      <div className="flex items-center gap-4 flex-1 justify-center">
        {/* Asset selector */}
        <div className="relative">
          <button
            onClick={() => setAssetOpen(v => !v)}
            className="flex items-center gap-2 bg-surface border border-card-border rounded-lg px-3 py-1.5 text-sm text-text-primary hover:border-brand-blue/50 transition-colors min-w-[160px]"
          >
            <span className="w-2 h-2 rounded-full bg-brand-blue inline-block" />
            <span className="font-mono font-medium">{currentAsset.ticker}</span>
            <span className="text-text-secondary text-xs">{currentAsset.name}</span>
            <ChevronDown size={14} className="ml-auto text-text-secondary" />
          </button>
          <AnimatePresence>
            {assetOpen && (
              <motion.div
                initial={{ opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                className="absolute top-full mt-1 left-0 bg-white border border-card-border rounded-xl overflow-hidden z-50 min-w-[180px] shadow-lg"
              >
                {ASSETS.map(a => (
                  <button
                    key={a.ticker}
                    onClick={() => { onAssetChange(a.ticker); setAssetOpen(false) }}
                    className={clsx(
                      'w-full flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-surface transition-colors',
                      a.ticker === selectedAsset ? 'text-text-primary bg-blue-50' : 'text-text-secondary'
                    )}
                  >
                    <span className={clsx(
                      'w-1.5 h-1.5 rounded-full',
                      a.ticker === selectedAsset ? 'bg-brand-blue' : 'bg-text-muted'
                    )} />
                    <span className="font-mono font-medium">{a.ticker}</span>
                    <span className="text-xs text-text-secondary">{a.name}</span>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Period pills */}
        <div className="flex items-center gap-1 bg-surface border border-card-border rounded-lg p-1">
          {PERIODS.map(p => (
            <button
              key={p.label}
              onClick={() => onPeriodChange(p)}
              className={clsx(
                'px-3 py-1 rounded-md text-xs font-mono font-medium transition-all',
                selectedPeriod.label === p.label
                  ? 'bg-brand-blue text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-hover'
              )}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Right */}
      <div className="flex items-center gap-4 ml-auto">
        {/* Market status */}
        <span className={clsx(
          'text-xs font-mono font-medium px-2.5 py-1 rounded-md border',
          isMarketOpen()
            ? 'text-risk-normal border-risk-normal/30 bg-green-50'
            : 'text-text-secondary border-card-border bg-surface'
        )}>
          {isMarketOpen() ? '● MARKET OPEN' : '○ MARKET CLOSED'}
        </span>

        {/* Last updated */}
        {lastUpdated && (
          <div className="text-right hidden md:block">
            <div className="text-text-muted text-[10px]">Last updated</div>
            <div className="text-text-secondary text-xs font-mono">{formatDate(lastUpdated, 'HH:mm:ss')}</div>
          </div>
        )}

        {/* Clock */}
        <div className="font-mono text-text-secondary text-sm tabular-nums hidden lg:block">
          {formatTime(time)}
        </div>

        {/* Refresh */}
        <button
          onClick={onRefresh}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 border border-card-border rounded-lg text-xs text-text-secondary hover:text-text-primary hover:border-brand-blue/50 transition-all disabled:opacity-50"
        >
          <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          <span className="hidden sm:inline">Refresh</span>
        </button>
      </div>
    </header>
  )
}
