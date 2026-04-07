import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { AlertCircle, TrendingDown, TrendingUp, Layers, Activity, Settings2, ShieldAlert } from 'lucide-react'
import AnomalyTable from '../components/widgets/AnomalyTable'
import AnomalyCalendar from '../components/widgets/AnomalyCalendar'
import {
  fetchCurrentAnalysis,
  fetchAdvancedAnomaly,
  fetchHistoricalAnomalies,
  fetchRegimeTimeline,
  fetchCrashLabels,
  fetchAnomalyMetrics,
  fetchThresholdAnalysis,
  fetchFalsePositives,
  fetchBubbleRisk,
} from '../services/api'
import { getRiskColor, getRiskLabel, getRiskTailwind } from '../utils/riskHelpers'
import { formatPrice, formatPct, formatScore, formatDate } from '../utils/formatters'
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea
} from 'recharts'
import clsx from 'clsx'

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
}
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
}

export default function Anomalies({ selectedAsset, loading: globalLoading }) {
  const [data, setData] = useState({
    current: null,
    advanced: null,
    historical: null,
    regime: null,
    crashLabels: null,
    metrics: null,
    thresholdAnalysis: null,
    falsePositives: null,
    bubbleRisk: null,
  })
  const [localLoading, setLocalLoading] = useState(true)
  const [error, setError] = useState(null)
  
  const [threshold, setThreshold] = useState(60)
  const [debouncedThreshold, setDebouncedThreshold] = useState(60)
  const [selectedEvent, setSelectedEvent] = useState(null)
  const [timeRange, setTimeRange] = useState('1Y') // 3M, 6M, 1Y, MAX

  // Debounce threshold
  useEffect(() => {
    const t = setTimeout(() => setDebouncedThreshold(threshold), 400)
    return () => clearTimeout(t)
  }, [threshold])

  useEffect(() => {
    // Reset selection on asset change
    setSelectedEvent(null)
  }, [selectedAsset])

  useEffect(() => {
    async function load() {
      if (!selectedAsset) return
      setLocalLoading(true)
      setError(null)
      try {
        const [curr, adv, hist, reg, labels, metrics, thresholdRec, falsePositives, bubbleRisk] = await Promise.allSettled([
          fetchCurrentAnalysis(selectedAsset),
          fetchAdvancedAnomaly(selectedAsset),
          fetchHistoricalAnomalies(selectedAsset, 100, debouncedThreshold),
          fetchRegimeTimeline(selectedAsset),
          fetchCrashLabels(),
          fetchAnomalyMetrics(selectedAsset, debouncedThreshold, 'ensemble_score', 7),
          fetchThresholdAnalysis(selectedAsset, {
            model: 'ensemble_score',
            minThreshold: 40,
            maxThreshold: 80,
            step: 2,
            costFp: 1,
            costFn: 5,
            windowDays: 7,
          }),
          fetchFalsePositives(selectedAsset, debouncedThreshold, 80, 'ensemble_score', 7),
          fetchBubbleRisk(selectedAsset),
        ])

        setData({
          current: curr.status === 'fulfilled' ? curr.value : null,
          advanced: adv.status === 'fulfilled' ? adv.value : null,
          historical: hist.status === 'fulfilled' ? hist.value : null,
          regime: reg.status === 'fulfilled' ? reg.value : null,
          crashLabels: labels.status === 'fulfilled' ? labels.value : null,
          metrics: metrics.status === 'fulfilled' ? metrics.value : null,
          thresholdAnalysis: thresholdRec.status === 'fulfilled' ? thresholdRec.value : null,
          falsePositives: falsePositives.status === 'fulfilled' ? falsePositives.value : null,
          bubbleRisk: bubbleRisk.status === 'fulfilled' ? bubbleRisk.value : null,
        })
      } catch (err) {
        setError('Failed to fetch anomaly data.')
      } finally {
        setLocalLoading(false)
      }
    }
    load()
  }, [selectedAsset, debouncedThreshold])

  const isLoading = globalLoading || localLoading

  // Extraction logic
  const currentPrice = data.advanced?.current_price ?? data.current?.price ?? 0
  const baselineScore = data.current?.ensemble_score ?? 0
  const advancedScore = data.advanced?.advanced_ensemble ?? baselineScore
  const delta = advancedScore - baselineScore
  const regimeStr = data.advanced?.current_regime ?? 'unknown'

  // Chart data filtering mapping based on timeRange
  const fullChartData = data.historical?.chart_data ?? []
  const filteredChartData = useMemo(() => {
    if (fullChartData.length === 0) return []
    if (timeRange === 'MAX') return fullChartData
    
    let days = 365
    if (timeRange === '3M') days = 90
    if (timeRange === '6M') days = 180
    
    // Naively assuming 1 item = 1 trading day in array ordering
    return fullChartData.slice(-days)
  }, [fullChartData, timeRange])

  // Derive continuous anomaly windows dynamically from filtered dataset
  const anomalyWindows = useMemo(() => {
    const windows = []
    let currentWindow = null

    for (let i = 0; i < filteredChartData.length; i++) {
      const row = filteredChartData[i]
      const isAnom = row.is_anomaly ?? (row.ensemble_score >= debouncedThreshold)
      
      if (isAnom) {
        if (!currentWindow) {
          currentWindow = { start: row.date, end: row.date }
        } else {
          currentWindow.end = row.date
        }
      } else {
        if (currentWindow) {
          windows.push(currentWindow)
          currentWindow = null
        }
      }
    }
    if (currentWindow) windows.push(currentWindow)
    return windows
  }, [filteredChartData, debouncedThreshold])

  const events = data.historical?.events ?? []
  const crashEvents = data.crashLabels?.events ?? []
  const falsePositiveEvents = data.falsePositives?.events ?? []

  const crashMarkers = useMemo(() => {
    if (!filteredChartData.length || !crashEvents.length) return []

    const chartDates = filteredChartData.map((row) => row.date)
    const firstTs = new Date(chartDates[0]).getTime()
    const lastTs = new Date(chartDates[chartDates.length - 1]).getTime()
    const maxDistanceMs = 3 * 24 * 60 * 60 * 1000
    const seen = new Set()

    return crashEvents
      .filter((event) => (event.assets_affected ?? []).includes(selectedAsset))
      .map((event) => {
        const eventTs = new Date(event.date).getTime()
        if (!Number.isFinite(eventTs)) return null
        if (eventTs < (firstTs - maxDistanceMs) || eventTs > (lastTs + maxDistanceMs)) return null

        let mappedDate = null
        let bestDistance = Number.POSITIVE_INFINITY

        chartDates.forEach((date) => {
          const ts = new Date(date).getTime()
          const distance = Math.abs(ts - eventTs)
          if (distance < bestDistance) {
            bestDistance = distance
            mappedDate = date
          }
        })

        if (!mappedDate || bestDistance > maxDistanceMs) return null
        if (seen.has(mappedDate)) return null
        seen.add(mappedDate)

        return {
          ...event,
          mappedDate,
        }
      })
      .filter(Boolean)
  }, [filteredChartData, crashEvents, selectedAsset])

  const falsePositiveMarkers = useMemo(() => {
    if (!filteredChartData.length || !falsePositiveEvents.length) return []
    const dateSet = new Set(filteredChartData.map((row) => row.date))
    return falsePositiveEvents.filter((event) => dateSet.has(event.date)).slice(0, 20)
  }, [filteredChartData, falsePositiveEvents])

  const bestThreshold = data.thresholdAnalysis?.best?.threshold
  const thresholdPrecision = data.thresholdAnalysis?.best?.precision
  const thresholdRecall = data.thresholdAnalysis?.best?.recall
  const thresholdF1 = data.thresholdAnalysis?.best?.f1
  const leadDistribution = data.metrics?.lead_time?.distribution ?? {}
  const bubbleRisk = data.bubbleRisk?.bubble_risk
  const bubbleLabel = data.bubbleRisk?.risk_label
  const bubbleWatch = data.bubbleRisk?.is_bubble_watch
  const aucPr = data.metrics?.auc_pr
  const brier = data.metrics?.brier_score

  // Dual Chart Custom Tooltip
  const DualChartTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const pPrice = payload.find(p => p.dataKey === 'close')
      const pScore = payload.find(p => p.dataKey === 'ensemble_score')
      const scoreVal = pScore?.value ?? 0
      const isAbove = scoreVal >= debouncedThreshold

      return (
        <div className="rounded-lg border border-card-border bg-card-bg p-3 shadow-xl min-w-[180px]">
          <p className="mb-2 text-xs font-semibold text-text-secondary">{formatDate(label)}</p>
          <div className="space-y-1.5 font-mono text-sm">
            <div className="flex justify-between items-center gap-4">
              <span className="text-text-muted">Price:</span>
              <span className="text-text-primary font-bold">{pPrice ? formatPrice(pPrice.value) : '—'}</span>
            </div>
            <div className="flex justify-between items-center gap-4">
              <span className="text-text-muted">Score:</span>
              <span style={{ color: getRiskColor(scoreVal) }} className="font-bold">
                {formatScore(scoreVal)}
              </span>
            </div>
          </div>
          {isAbove && (
             <div className="mt-2 text-[10px] font-bold text-red-600 bg-red-500/10 px-2 py-1 rounded inline-flex uppercase">
               Above Threshold
             </div>
          )}
        </div>
      )
    }
    return null
  }

  return (
    <motion.div 
      className="space-y-8 max-w-[1600px] mx-auto w-full pb-12"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      {/* SECTION 1 - OVERVIEW HEADER */}
      <motion.div variants={itemVariants}>
        <div className="mb-6">
          <h1 className="text-2xl md:text-3xl font-bold text-text-primary flex items-center gap-3">
            <ShieldAlert className="text-brand-blue" />
            Anomaly Detection Command Center
          </h1>
          <p className="text-text-secondary mt-2">
            Ensemble structural stress detection & regime analysis for <strong className="text-text-primary">{selectedAsset}</strong>.
          </p>
        </div>

        {error && (
          <div className="bg-red-50 text-red-600 p-4 rounded-lg flex items-center gap-2 text-sm font-medium mb-6">
            <AlertCircle size={18} /> {error}
          </div>
        )}

        <div className="flex flex-col xl:flex-row gap-4 mb-4">
          {/* KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 flex-1">
            <Card className="p-4 flex flex-col justify-center">
              <span className="text-text-muted text-xs uppercase tracking-wider mb-1">Baseline Score</span>
              <div className="flex items-end gap-2">
                <span className={clsx("text-2xl font-bold font-mono", getRiskTailwind(baselineScore))}>
                  {isLoading ? '--' : formatScore(baselineScore)}
                </span>
                {!isLoading && <Badge variant="outline" className="mb-1">{getRiskLabel(baselineScore)}</Badge>}
              </div>
            </Card>

            <Card className="p-4 flex flex-col justify-center border-brand-blue/30 shadow-sm relative overflow-hidden">
               <div className="absolute top-0 right-0 p-2 opacity-10">
                 <Settings2 size={40} />
               </div>
              <span className="text-text-muted text-xs uppercase tracking-wider mb-1 relative">Adv. Score</span>
              <div className="flex items-end gap-2 relative">
                <span className={clsx("text-2xl font-bold font-mono", getRiskTailwind(advancedScore))}>
                  {isLoading ? '--' : formatScore(advancedScore)}
                </span>
                {!isLoading && <Badge variant="outline" className="mb-1">{getRiskLabel(advancedScore)}</Badge>}
              </div>
            </Card>

            <Card className="p-4 flex flex-col justify-center">
              <span className="text-text-muted text-xs uppercase tracking-wider mb-1">Model Delta</span>
               <div className="flex items-center gap-2">
                 <span className={clsx("text-xl font-bold font-mono", delta > 0 ? "text-red-500" : delta < 0 ? "text-emerald-500" : "text-text-primary")}>
                   {isLoading ? '--' : `${delta > 0 ? '+' : ''}${formatScore(delta)}`}
                 </span>
                 {!isLoading && delta !== 0 && (delta > 0 ? <TrendingUp size={16} className="text-red-500"/> : <TrendingDown size={16} className="text-emerald-500"/>)}
               </div>
            </Card>

             <Card className="p-4 flex flex-col justify-center">
              <span className="text-text-muted text-xs uppercase tracking-wider mb-1">Current Price</span>
               <div className="flex items-center gap-2">
                 <span className="text-xl font-bold font-mono text-text-primary">
                   {isLoading ? '--' : formatPrice(currentPrice)}
                 </span>
               </div>
            </Card>

            <Card className="p-4 flex flex-col justify-center">
              <span className="text-text-muted text-xs uppercase tracking-wider mb-1">Regime context</span>
               <div className="flex items-center gap-2">
                 <Layers size={18} className="text-text-muted" />
                 <span className="text-sm font-semibold capitalize text-brand-blue">
                   {isLoading ? '--' : regimeStr}
                 </span>
               </div>
            </Card>
          </div>

          {/* Threshold Control */}
          <Card className="p-4 w-full xl:w-72 flex flex-col justify-center shrink-0 bg-surface/50">
             <div className="flex justify-between items-center mb-2">
               <span className="text-sm font-bold text-text-primary">Detection Threshold</span>
               <span className="font-mono text-sm bg-brand-blue text-white px-2 py-0.5 rounded">
                 {threshold}
               </span>
             </div>
             <input 
               type="range" 
               min="40" 
               max="80" 
               value={threshold} 
               onChange={(e) => setThreshold(Number(e.target.value))}
               className="w-full accent-brand-blue mb-2"
             />
             <p className="text-[10px] text-text-muted leading-tight">Controls what is considered an anomaly in historical charts. Triggers data refetch when changed.</p>
          </Card>
        </div>
      </motion.div>

      {/* SECTION 2 - DUAL TIMELINE CHART */}
      <motion.div variants={itemVariants}>
        <Card className="p-5 flex flex-col shadow-sm border border-card-border">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
             <div>
               <h2 className="text-lg font-bold text-text-primary flex items-center gap-2">
                 <Activity size={18} className="text-brand-blue" /> Structural Stress Timeline
               </h2>
               <p className="text-xs text-text-muted mt-1">
                 Compare price action (left axis) directly against the multi-model ensemble anomaly score (right axis).
               </p>
             </div>
             
             {/* Time range selector */}
             <div className="flex bg-surface p-1 rounded-lg border border-card-border shrink-0">
                {['3M', '6M', '1Y', 'MAX'].map(tr => (
                  <button
                    key={tr}
                    onClick={() => setTimeRange(tr)}
                    className={clsx(
                      "px-3 py-1 text-xs font-semibold rounded-md transition-colors",
                      timeRange === tr ? "bg-card-bg text-brand-blue shadow-sm" : "text-text-secondary hover:text-text-primary"
                    )}
                  >
                    {tr}
                  </button>
                ))}
             </div>
          </div>

          <div className="h-[400px] w-full relative">
            {isLoading && (
              <div className="absolute inset-0 z-10 bg-card-bg/50 backdrop-blur-[2px] flex items-center justify-center rounded border border-card-border">
                 <span className="text-sm font-medium text-text-muted animate-pulse">Loading timeline...</span>
              </div>
            )}
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={filteredChartData} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
                <defs>
                   <linearGradient id="anomArea" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#EF4444" stopOpacity={0.2}/>
                      <stop offset="95%" stopColor="#EF4444" stopOpacity={0.0}/>
                   </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EAEBEE" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={tick => formatDate(tick, 'MMM d')} 
                  tick={{ fontSize: 11, fill: '#6B7280' }} 
                  axisLine={false} 
                  tickLine={false} 
                  minTickGap={30}
                  dy={10}
                />
                
                {/* Price Left Axis */}
                <YAxis 
                  yAxisId="left" 
                  domain={['auto', 'auto']}
                  tickFormatter={formatPrice}
                  tick={{ fontSize: 11, fill: '#6B7280' }}
                  axisLine={false}
                  tickLine={false}
                  width={60}
                />
                
                {/* Score Right Axis */}
                <YAxis 
                  yAxisId="right" 
                  orientation="right" 
                  domain={[0, 100]} 
                  tick={{ fontSize: 11, fill: '#EF4444' }}
                  axisLine={false}
                  tickLine={false}
                  width={40}
                />

                {/* Shaded event regions */}
                {anomalyWindows.map((win, idx) => (
                   <ReferenceArea
                      key={idx}
                      yAxisId="left"
                      x1={win.start}
                      x2={win.end}
                      fill="#EF4444"
                      fillOpacity={0.15}
                   />
                ))}

                {crashMarkers.map((event) => (
                  <ReferenceLine
                    key={`crash-${event.mappedDate}`}
                    x={event.mappedDate}
                    stroke="#B91C1C"
                    strokeDasharray="2 3"
                    strokeWidth={event.impact === 'extreme' ? 2 : 1}
                  />
                ))}

                {falsePositiveMarkers.map((event) => (
                  <ReferenceLine
                    key={`fp-${event.date}`}
                    x={event.date}
                    stroke="#D97706"
                    strokeDasharray="1 4"
                    strokeWidth={1}
                  />
                ))}

                <RechartsTooltip content={<DualChartTooltip />} cursor={{ fill: '#F5F6F8' }} />
                
                {/* Dynamic threshold marker line on Right Axis */}
                <ReferenceLine 
                  yAxisId="right" 
                  y={debouncedThreshold} 
                  stroke="#F59E0B" 
                  strokeDasharray="4 4" 
                  label={{ position: 'insideTopLeft', value: `Threshold (${debouncedThreshold})`, fill: '#F59E0B', fontSize: 10 }}
                />

                <Area 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="close" 
                  fill="url(#anomArea)" 
                  stroke="#2563EB" 
                  strokeWidth={2}
                  activeDot={{ r: 4, strokeWidth: 0, fill: '#2563EB' }}
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="ensemble_score" 
                  stroke="#EF4444" 
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 4, fill: '#EF4444', stroke: '#fff', strokeWidth: 2 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-4 text-[11px] text-text-secondary">
            <span className="inline-flex items-center gap-1.5">
              <span className="h-px w-5 bg-[#B91C1C]" /> Crash label marker
            </span>
            <span className="inline-flex items-center gap-1.5">
              <span className="h-px w-5 border-t border-dashed border-[#D97706]" /> False positive marker
            </span>
            <span>Crash markers shown: {crashMarkers.length}</span>
            <span>False positives shown: {falsePositiveMarkers.length}</span>
          </div>
        </Card>
      </motion.div>

      <motion.div variants={itemVariants}>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <Card className="p-5">
            <h3 className="text-sm font-bold uppercase tracking-[0.12em] text-text-primary">Bubble Risk Monitor</h3>
            <div className="mt-3 flex items-end justify-between gap-2">
              <div className="text-3xl font-mono font-bold" style={{ color: getRiskColor(bubbleRisk ?? 0) }}>
                {bubbleRisk != null ? formatScore(bubbleRisk) : '--'}
              </div>
              {bubbleLabel && <Badge variant={bubbleWatch ? 'orange' : 'green'}>{bubbleLabel}</Badge>}
            </div>
            <div className="mt-3 text-xs text-text-secondary">
              Bubble score: <span className="font-mono text-text-primary">{data.bubbleRisk?.bubble_score ?? '--'}</span>
            </div>
            <div className="mt-1 text-xs text-text-secondary">
              Percentile: <span className="font-mono text-text-primary">{data.bubbleRisk?.bubble_percentile != null ? `${(data.bubbleRisk.bubble_percentile * 100).toFixed(1)}%` : '--'}</span>
            </div>
          </Card>

          <Card className="p-5">
            <h3 className="text-sm font-bold uppercase tracking-[0.12em] text-text-primary">Threshold Quality</h3>
            <div className="mt-3 text-xs text-text-secondary">
              Recommended threshold: <span className="font-mono text-text-primary">{bestThreshold ?? '--'}</span>
            </div>
            <div className="mt-1 text-xs text-text-secondary">
              Precision: <span className="font-mono text-text-primary">{thresholdPrecision != null ? formatPct(thresholdPrecision * 100) : '--'}</span>
            </div>
            <div className="mt-1 text-xs text-text-secondary">
              Recall: <span className="font-mono text-text-primary">{thresholdRecall != null ? formatPct(thresholdRecall * 100) : '--'}</span>
            </div>
            <div className="mt-1 text-xs text-text-secondary">
              F1: <span className="font-mono text-text-primary">{thresholdF1 != null ? thresholdF1.toFixed(3) : '--'}</span>
            </div>
            <div className="mt-3 border-t border-card-border pt-3 text-xs text-text-secondary">
              AUCPR: <span className="font-mono text-text-primary">{aucPr != null ? aucPr.toFixed(3) : '--'}</span>
              {' '}| Brier: <span className="font-mono text-text-primary">{brier != null ? brier.toFixed(4) : '--'}</span>
            </div>
          </Card>

          <Card className="p-5">
            <h3 className="text-sm font-bold uppercase tracking-[0.12em] text-text-primary">Lead Time and False Positives</h3>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-text-secondary">
              <div>1-3d early: <span className="font-mono text-text-primary">{leadDistribution.early_1_3 ?? 0}</span></div>
              <div>4-7d early: <span className="font-mono text-text-primary">{leadDistribution.early_4_7 ?? 0}</span></div>
              <div>{'>'}7d early: <span className="font-mono text-text-primary">{leadDistribution.early_gt_7 ?? 0}</span></div>
              <div>Missed: <span className="font-mono text-text-primary">{leadDistribution.missed ?? 0}</span></div>
            </div>
            <div className="mt-3 border-t border-card-border pt-3 text-xs text-text-secondary">
              False positives (threshold {debouncedThreshold}):
              <span className="ml-2 font-mono text-text-primary">{data.falsePositives?.count ?? 0}</span>
            </div>
            <div className="mt-2 text-xs text-text-muted">
              Recent dates: {falsePositiveEvents.slice(0, 3).map((e) => e.date).join(', ') || 'None'}
            </div>
          </Card>
        </div>
      </motion.div>

      {/* SECTION 3 - EVENTS */}
      <motion.div variants={itemVariants}>
         {/* LEFT / TABLE */}
         <Card className="flex flex-col shadow-sm border border-card-border overflow-hidden h-[420px]">
           <div className="p-5 border-b border-card-border bg-surface/30">
              <h3 className="font-bold text-text-primary">Top Anomaly Windows</h3>
            <p className="text-xs text-text-muted mt-1">Detected high-confidence structural anomaly windows.</p>
           </div>
           <div className="flex-1 overflow-y-auto custom-scrollbar">
              <AnomalyTable 
                events={events} 
                loading={isLoading} 
                maxRows={12} 
                selectedEvent={selectedEvent}
                onRowClick={setSelectedEvent}
              />
           </div>
         </Card>
      </motion.div>

      {/* SECTION 4 - HEATMAP */}
      <motion.div variants={itemVariants}>
        <h3 className="font-bold text-text-primary mb-1">Density Calendar</h3>
        <p className="text-xs text-text-muted mb-4">
          Historical heatmap of anomaly occurrences for pattern recurrence.
        </p>
        <div className="flex-1 flex flex-col">
          <AnomalyCalendar events={events} loading={isLoading} />
        </div>
      </motion.div>

      {/* SECTION 6 - HOW DETECTION WORKS (MUST BE LAST) */}
      <motion.div variants={itemVariants} className="pt-8">
        <h3 className="font-bold text-text-primary mb-6 text-center text-xl">How Detection Works</h3>
        
        <div className="grid grid-cols-2 lg:grid-cols-7 gap-2 items-center mb-8 bg-surface/30 p-6 rounded-2xl border border-card-border/50">
           {/* Steps */}
           {[
             { title: "Ingest", sub: "OHLCV + Webhooks", num: 1 },
             { title: "Features", sub: "Vol & Drawdown", num: 2 },
             { title: "Baseline", sub: "Z-score / Forest", num: 3 },
             { title: "Advanced", sub: "XGB, TCN, HMM", num: 4 },
             { title: "Ensemble", sub: "Weighted (0-100)", num: 5 },
             { title: "Windows", sub: "Threshold grouping", num: 6 },
             { title: "Alerts", sub: "Context via Regime", num: 7 }
           ].map((step, i) => (
             <div key={i} className="flex flex-col items-center text-center relative group">
                <div className="w-8 h-8 rounded-full bg-card-bg border border-card-border text-brand-blue font-bold text-xs flex items-center justify-center mb-2 z-10 group-hover:scale-110 transition-transform shadow-sm">
                  {step.num}
                </div>
                <span className="font-medium text-xs text-text-primary leading-tight">{step.title}</span>
                <span className="text-[9px] text-text-muted uppercase tracking-wider mt-1">{step.sub}</span>
                {i < 6 && (
                  <div className="hidden lg:block absolute top-[15px] left-[60%] w-[80%] h-px bg-card-border -z-0" />
                )}
             </div>
           ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
           <Card className="p-4 bg-transparent border-t-2 border-t-brand-blue border-r-0 border-b-0 border-l-0 rounded-none">
             <strong className="block mb-1">Anomaly Score</strong>
             <span className="text-text-muted text-xs">A unified probability 0–100 derived from combining statistical checks, deep learning decoders, and boosting outputs.</span>
           </Card>
           <Card className="p-4 bg-transparent border-t-2 border-t-[#F59E0B] border-r-0 border-b-0 border-l-0 rounded-none">
             <strong className="block mb-1">Threshold Validation</strong>
             <span className="text-text-muted text-xs">Scores above a dynamic threshold (default 60) are converted definitively into an active anomalous state.</span>
           </Card>
           <Card className="p-4 bg-transparent border-t-2 border-t-[#EF4444] border-r-0 border-b-0 border-l-0 rounded-none">
             <strong className="block mb-1">Detection Window</strong>
             <span className="text-text-muted text-xs">Contiguous periods of anomalous scores are rolled into logical timeline windows to denote sustained structural stress.</span>
           </Card>
        </div>
      </motion.div>

    </motion.div>
  )
}
