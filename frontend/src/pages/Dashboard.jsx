import { useEffect, useMemo, useState } from 'react'
import { Activity, Layers, Sparkles, Zap, RefreshCw, CheckCircle2, ChevronRight, BarChart3, AlertCircle } from 'lucide-react'
import clsx from 'clsx'
import { motion, AnimatePresence } from 'framer-motion'
import { Card } from '../components/ui/Card'

import AnomalyTable from '../components/widgets/AnomalyTable'
import PriceAreaChart from '../components/charts/PriceAreaChart'
import RiskScoreChart from '../components/charts/RiskScoreChart'
import EvaluationSnapshot from '../components/widgets/EvaluationSnapshot'
import { formatPct, formatPrice, formatScore } from '../utils/formatters'
import { getRiskColor, getRiskLabel } from '../utils/riskHelpers'
import { API_BASE, ASSETS } from '../constants/config'
import { COLOURS } from '../constants/colours'

const ASSET_NAMES = {
  SP500: 'S&P 500',
  VIX: 'Volatility Index',
  BTC: 'Bitcoin',
  GOLD: 'Gold',
  NASDAQ: 'Nasdaq 100',
  TESLA: 'Tesla',
}

function riskBadgeClasses(score) {
  if (score >= 75) return 'bg-red-50 text-red-700 border-red-200'
  if (score >= 60) return 'bg-orange-50 text-orange-700 border-orange-200'
  if (score >= 40) return 'bg-amber-50 text-amber-700 border-amber-200'
  return 'bg-emerald-50 text-emerald-700 border-emerald-200'
}

function Sparkline({ values = [], selected = false }) {
  if (!values.length) {
    return <div className="h-12 rounded-lg bg-surface/50" />
  }

  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100
      const y = 100 - (((value - min) / range) * 76 + 12)
      return `${x},${y}`
    })
    .join(' ')

  const strokeColor = selected ? '#2563EB' : '#9CA3AF'

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-12 w-full overflow-visible">
      <defs>
        <linearGradient id={`sparkFill-${selected}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={strokeColor} stopOpacity={selected ? "0.3" : "0.1"} />
          <stop offset="100%" stopColor={strokeColor} stopOpacity="0.0" />
        </linearGradient>
      </defs>
      <polyline
        points={points}
        fill="none"
        stroke={strokeColor}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polygon points={`0,100 ${points} 100,100`} fill={`url(#sparkFill-${selected})`} />
    </svg>
  )
}

export default function Dashboard({ 
  current, historical, priceForecast, evaluation, summary,
  loading, selectedAsset, onSelectAsset, apiOnline, lastUpdated, onRefresh
}) {
  const [advancedData, setAdvancedData] = useState(null)
  const [overviewCards, setOverviewCards] = useState([])
  const [overviewLoading, setOverviewLoading] = useState(true)
  const [showWindows, setShowWindows] = useState(true)

  // Stagger motion
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 }
    }
  }
  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0, transition: { duration: 0.4 } }
  }

  useEffect(() => {
    if (!selectedAsset) return
    fetch(`${API_BASE}/anomaly/advanced/${selectedAsset}`)
      .then((res) => res.json())
      .then(setAdvancedData)
      .catch(() => setAdvancedData(null))
  }, [selectedAsset])

  useEffect(() => {
    let active = true

    const loadOverview = async () => {
      setOverviewLoading(true)
      try {
        const summaryAssets = Array.isArray(summary?.assets) ? summary.assets : []
        if (!summaryAssets.length) {
          // If summary not passed down, fetch it
          const res = await fetch(`${API_BASE}/summary`)
          const json = await res.json()
          if (json.assets) summaryAssets.push(...json.assets)
        }

        const cards = await Promise.all(summaryAssets.map(async (item) => {
          if (item?.error) {
            return {
              asset: item.asset,
              name: ASSET_NAMES[item.asset] ?? item.asset,
              currentPrice: null,
              score: 0,
              riskLabel: 'Unavailable',
              delta30dPct: 0,
              sparkline: [],
            }
          }

          const [forecastRes, historyRes] = await Promise.allSettled([
            fetch(`${API_BASE}/forecast/price/${item.asset}?horizon=30&method=auto`).then((r) => r.json()),
            fetch(`${API_BASE}/anomaly/historical/${item.asset}?top_n=30`).then((r) => r.json()),
          ])

          const forecastJson = forecastRes.status === 'fulfilled' ? forecastRes.value : null
          const historyJson = historyRes.status === 'fulfilled' ? historyRes.value : null

          const forecast30d =
            forecastJson?.summary?.forecast_30d ??
            forecastJson?.forecast?.values?.[forecastJson?.forecast?.values?.length - 1] ??
            item?.forecast_1d ??
            item?.current_price ??
            0

          const currentPrice = item?.current_price ?? 0
          const delta30dPct = currentPrice > 0
            ? ((forecast30d / currentPrice) - 1) * 100
            : 0

          return {
            asset: item.asset,
            name: ASSET_NAMES[item.asset] ?? item.asset,
            currentPrice,
            score: item?.anomaly_score ?? 0,
            riskLabel: item?.risk_label ?? getRiskLabel(item?.anomaly_score),
            delta30dPct,
            sparkline: (historyJson?.chart_data ?? [])
              .slice(-30)
              .map((row) => row?.close)
              .filter((value) => Number.isFinite(value)),
          }
        }))

        const ordered = ASSETS.map((asset) => cards.find((card) => card.asset === asset.ticker)).filter(Boolean)
        if (active) setOverviewCards(ordered)
      } catch {
        if (active) setOverviewCards([])
      } finally {
        if (active) setOverviewLoading(false)
      }
    }

    loadOverview()
    return () => {
      active = false
    }
  }, [selectedAsset, summary])

  const score = current?.ensemble_score ?? 0
  const advScore = advancedData?.advanced_ensemble ?? score
  const regime = advancedData?.current_regime ?? 'unknown'

  const currentPrice = priceForecast?.current_price ?? current?.price ?? 0
  const chartData = historical?.chart_data ?? []
  
  // Calculate 1D Change
  let oneDayChangePct = 0
  if (chartData.length >= 2) {
    const today = chartData[chartData.length - 1].close
    const yesterday = chartData[chartData.length - 2].close
    if (yesterday > 0) oneDayChangePct = ((today / yesterday) - 1) * 100
  }

  const anomalyPts = useMemo(
    () => (historical?.events ?? []).slice(0, 40).map((event) => ({ date: event.date, start_date: event.start_date, end_date: event.end_date, close: null })),
    [historical]
  )

  const selectedFullName = ASSET_NAMES[selectedAsset] || selectedAsset

  return (
    <motion.div 
      className="space-y-6 max-w-[1600px] mx-auto w-full"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      {/* 1) Command Bar / Hero */}
      <motion.div variants={itemVariants}>
        <Card className="relative overflow-hidden border-brand-blue/15 bg-card-bg shadow-sm p-4 md:p-6">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_100%_0%,rgba(37,99,235,0.08),transparent_25%)] pointer-events-none" />
          <div className="relative flex flex-col md:flex-row md:items-center justify-between gap-4">
            
            <div className="flex items-center gap-4">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-text-primary flex items-center gap-2">
                  <Activity className="text-brand-blue shrink-0" size={24} />
                  {selectedFullName} 
                  <span className="text-text-muted font-normal text-xl">({selectedAsset})</span>
                </h1>
              </div>
              <div className="hidden sm:flex self-stretch w-px bg-card-border mx-2" />
              <div className="hidden sm:flex">
                 <span className={clsx('rounded-full border px-2.5 py-1 text-xs font-bold uppercase tracking-widest', riskBadgeClasses(score))}>
                    {getRiskLabel(score)}
                 </span>
              </div>
            </div>

            <div className="flex items-center justify-between md:justify-end gap-4 w-full md:w-auto">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-xs font-mono text-text-secondary">
                  <span className={clsx("w-2 h-2 rounded-full animate-pulse", apiOnline ? "bg-emerald-500" : "bg-red-500")} />
                  {apiOnline ? 'Live' : 'Offline'}
                </div>
                {lastUpdated && (
                  <span className="hidden lg:block text-xs text-text-muted whitespace-nowrap">
                    Updated: {lastUpdated.toLocaleTimeString()}
                  </span>
                )}
                <button 
                  onClick={onRefresh}
                  disabled={loading}
                  className="p-2 hover:bg-surface rounded-md text-text-secondary hover:text-brand-blue transition-colors disabled:opacity-50 border border-transparent hover:border-card-border"
                >
                  <RefreshCw size={16} className={clsx(loading && "animate-spin")} />
                </button>
              </div>
            </div>

          </div>
        </Card>
      </motion.div>

      {/* 2) Asset Overview Cards (Interactive) */}
      <motion.div variants={itemVariants} className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
        {(overviewLoading ? Array.from({ length: 6 }) : overviewCards).map((card, index) => {
          const isSkeleton = overviewLoading || !card
          const isSelected = !isSkeleton && card.asset === selectedAsset

          if (isSkeleton) {
            return (
              <div key={`skeleton-${index}`} className="animate-pulse flex flex-col p-4 rounded-xl bg-card-bg border border-card-border h-[130px]">
                 <div className="h-3 w-12 bg-surface rounded mb-2" />
                 <div className="h-5 w-20 bg-surface rounded mb-4" />
                 <div className="h-10 w-full bg-surface rounded mt-auto" />
              </div>
            )
          }

          return (
            <button
              key={card.asset}
              onClick={() => onSelectAsset && onSelectAsset(card.asset)}
              title="Click to focus"
              className={clsx(
                "group relative overflow-hidden text-left bg-card-bg border rounded-xl p-4 transition-all duration-300 outline-none focus-visible:ring-2 focus-visible:ring-brand-blue focus-visible:ring-offset-2 hover:-translate-y-0.5",
                isSelected 
                  ? "border-brand-blue shadow-md shadow-brand-blue/10" 
                  : "border-card-border hover:border-brand-blue/40 shadow-sm"
              )}
            >
              {/* Selected indicator */}
              <AnimatePresence>
                {isSelected && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                    className="absolute top-3 right-3 text-brand-blue"
                  >
                    <CheckCircle2 size={16} className="fill-blue-50 text-brand-blue" />
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="flex justify-between items-start mb-2">
                <div>
                  <div className={clsx("text-xs font-bold uppercase tracking-widest transition-colors", isSelected ? "text-brand-blue" : "text-text-muted group-hover:text-text-secondary")}>
                    {card.asset}
                  </div>
                  <div className="mt-1 font-mono text-lg font-bold text-text-primary leading-none">
                    {formatPrice(card.currentPrice)}
                  </div>
                </div>
              </div>

              <div className="mt-4 opacity-80 group-hover:opacity-100 transition-opacity">
                <Sparkline values={card.sparkline} selected={isSelected} />
              </div>
            </button>
          )
        })}
      </motion.div>

      {/* 3) KPI Strip */}
      <motion.div variants={itemVariants} className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {[
          { label: 'Current Price', value: formatPrice(currentPrice), isMono: true },
          { label: '1D Change', value: `${oneDayChangePct > 0 ? '+' : ''}${formatPct(oneDayChangePct)}`, isMono: true, color: oneDayChangePct >= 0 ? COLOURS.riskNormal : COLOURS.riskHigh },
          { label: 'Baseline Score', value: formatScore(score), isMono: true, color: getRiskColor(score) },
          { label: 'Advanced Score', value: advScore ? formatScore(advScore) : '...', isMono: true, color: advScore ? getRiskColor(advScore) : undefined },
          { label: 'Regime', value: regime, isMono: false, color: regime.toLowerCase().includes('bear') || regime.toLowerCase().includes('crisis') ? COLOURS.riskHigh : COLOURS.textPrimary },
        ].map((kpi, i) => (
          <div key={i} className="bg-card-bg border border-card-border rounded-xl p-4 shadow-sm flex flex-col justify-center">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-1">{kpi.label}</span>
            <span 
              className={clsx("text-2xl font-bold truncate", kpi.isMono ? "font-mono" : "font-sans capitalize")}
              style={{ color: kpi.color || COLOURS.textPrimary }}
            >
              {loading ? <span className="animate-pulse bg-surface text-transparent rounded w-16">...</span> : kpi.value}
            </span>
          </div>
        ))}
      </motion.div>

      <div className="space-y-6">
          {/* 4) Update PriceAreaChart */}
          <motion.div variants={itemVariants}>
            <Card>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
                <div>
                  <h2 className="text-lg font-bold text-text-primary flex items-center gap-2">
                    <BarChart3 size={18} className="text-brand-blue" />
                    Price & Anomaly Overlay
                  </h2>
                  <p className="text-xs text-text-secondary mt-1">Highlighted regions show periods of structural market divergence.</p>
                </div>
                <label className="flex items-center gap-2 text-sm text-text-primary cursor-pointer select-none bg-surface/50 border border-card-border px-3 py-1.5 rounded-lg hover:bg-surface transition-colors">
                  <input 
                    type="checkbox" 
                    className="rounded border-card-border text-brand-blue focus:ring-brand-blue"
                    checked={showWindows}
                    onChange={(e) => setShowWindows(e.target.checked)}
                  />
                  Show anomaly windows
                </label>
              </div>
              {loading ? (
                <div className="h-[280px] animate-pulse rounded-lg bg-surface flex items-center justify-center text-text-muted text-sm">Loading chart...</div>
              ) : (
                <PriceAreaChart data={chartData} anomalyPoints={anomalyPts} showWindows={showWindows} />
              )}
            </Card>
          </motion.div>

          {/* Risk Timeline */}
          <motion.div variants={itemVariants}>
            <Card>
              <h2 className="mb-1 text-lg font-bold text-text-primary flex items-center gap-2">
                <AlertCircle size={18} className="text-brand-blue" />
                Risk Score Trajectory
              </h2>
              <p className="mb-4 text-xs text-text-secondary">Historical ensemble anomaly score vs threshold (60.0).</p>
              {loading ? (
                <div className="h-[220px] animate-pulse rounded-lg bg-surface" />
              ) : (
                <RiskScoreChart data={chartData} />
              )}
            </Card>
          </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_250px] gap-6 pb-12">
        {/* Recent Anomalies */}
        <motion.div variants={itemVariants}>
          <Card className="h-full">
            <h2 className="mb-4 text-lg font-bold text-text-primary">Recent Anomaly Events</h2>
            <AnomalyTable events={historical?.events ?? []} loading={loading} maxRows={8} />
          </Card>
        </motion.div>

        {/* 7) Evaluation Snapshot */}
        <motion.div variants={itemVariants}>
          <Card className="flex flex-col h-full bg-gradient-to-b from-white to-surface/30">
            <h2 className="mb-4 text-sm font-bold uppercase tracking-widest text-text-primary border-b border-card-border pb-2">
              Model Evaluation
            </h2>
            <div className="flex-1">
              {loading ? (
                <div className="h-[150px] animate-pulse rounded-lg bg-surface w-full" />
              ) : (
                <EvaluationSnapshot evaluation={evaluation} selectedAsset={selectedAsset} />
              )}
            </div>
          </Card>
        </motion.div>
      </div>

    </motion.div>
  )
}
