import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import clsx from 'clsx'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { fetchAnomalyForecast } from '../../services/api'
import { formatDate, formatScore } from '../../utils/formatters'

const DAY_OPTIONS = [5, 10, 15, 30]
const MODE_OPTIONS = [
  { id: 'hybrid', label: 'Hybrid' },
  { id: 'dl', label: 'DL Composite' },
  { id: 'advanced', label: 'Advanced' },
  { id: 'ensemble', label: 'Baseline' },
]

const MODE_LINE_COLORS = {
  hybrid: '#1D6FDC',
  dl: '#0F766E',
  advanced: '#B45309',
  ensemble: '#475569',
}

function getRiskVariant(score) {
  if (score >= 75) return 'red'
  if (score >= 60) return 'orange'
  if (score >= 40) return 'yellow'
  return 'green'
}

function isMatchingForecast(data, asset, days, mode) {
  const dataMode = String(data?.mode || '').toLowerCase()
  return (
    data?.asset === asset
    && Number(data?.horizon) === Number(days)
    && dataMode === String(mode || '').toLowerCase()
    && Array.isArray(data?.forecast)
  )
}

function RiskTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  return (
    <div className="min-w-[200px] rounded-xl border border-card-border bg-white p-3 shadow-float">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-text-muted">
        {formatDate(label, 'MMM dd, yyyy')}
      </div>
      <div className="space-y-1 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-text-secondary">Score</span>
          <span className="font-mono font-bold text-text-primary">{formatScore(row?.score)}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-text-secondary">Lower</span>
          <span className="font-mono text-text-primary">{formatScore(row?.lower)}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-text-secondary">Upper</span>
          <span className="font-mono text-text-primary">{formatScore(row?.upper)}</span>
        </div>
      </div>
      <div className="mt-2 border-t border-card-border pt-2">
        <Badge variant={getRiskVariant(row?.score ?? 0)}>{row?.risk_label || 'Unknown'}</Badge>
      </div>
    </div>
  )
}

export default function AnomalyRiskForecastPanel({ asset, defaultDays = 10, initialForecast = null }) {
  const defaultMode = String(initialForecast?.mode || 'hybrid').toLowerCase()
  const [selectedDays, setSelectedDays] = useState(defaultDays)
  const [selectedMode, setSelectedMode] = useState(defaultMode)
  const [forecastData, setForecastData] = useState(
    isMatchingForecast(initialForecast, asset, defaultDays, defaultMode) ? initialForecast : null
  )
  const [comparisonByMode, setComparisonByMode] = useState({})
  const [comparisonLoading, setComparisonLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!asset) return

    if (isMatchingForecast(initialForecast, asset, selectedDays, selectedMode)) {
      setForecastData(initialForecast)
      setError(null)
      return
    }

    let active = true
    setLoading(true)
    setError(null)

    fetchAnomalyForecast(asset, selectedDays, { mode: selectedMode, method: 'hybrid' })
      .then((data) => {
        if (!active) return
        setForecastData(data)
      })
      .catch((err) => {
        if (!active) return
        setError(err?.message || 'Failed to load anomaly risk forecast')
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [asset, selectedDays, selectedMode, initialForecast])

  useEffect(() => {
    if (!asset) return

    let active = true
    setComparisonLoading(true)

    Promise.allSettled(
      MODE_OPTIONS.map((mode) =>
        fetchAnomalyForecast(asset, selectedDays, { mode: mode.id, method: 'hybrid' })
      )
    )
      .then((results) => {
        if (!active) return

        const next = {}
        results.forEach((result, idx) => {
          const modeId = MODE_OPTIONS[idx].id
          if (result.status === 'fulfilled' && Array.isArray(result.value?.forecast)) {
            next[modeId] = result.value
          }
        })
        setComparisonByMode(next)
      })
      .finally(() => {
        if (active) setComparisonLoading(false)
      })

    return () => {
      active = false
    }
  }, [asset, selectedDays])

  const rows = Array.isArray(forecastData?.forecast) ? forecastData.forecast : []
  const modeMeta = MODE_OPTIONS.find((item) => item.id === selectedMode)
  const modelsUsed = Array.isArray(forecastData?.models_used) ? forecastData.models_used : []
  const availableComparisonModes = MODE_OPTIONS.filter((mode) => Array.isArray(comparisonByMode[mode.id]?.forecast))
  const chartData = useMemo(
    () => rows.map((point) => ({
      date: point?.date,
      score: Number(point?.score) || 0,
      lower: Number(point?.lower ?? point?.score) || 0,
      upper: Number(point?.upper ?? point?.score) || 0,
      risk_label: point?.risk_label,
    })),
    [rows]
  )

  const comparisonChartData = useMemo(() => {
    const rowMap = new Map()

    MODE_OPTIONS.forEach((mode) => {
      const points = comparisonByMode[mode.id]?.forecast ?? []
      points.forEach((point) => {
        const date = point?.date
        if (!date) return

        if (!rowMap.has(date)) {
          rowMap.set(date, { date })
        }

        rowMap.get(date)[mode.id] = Number(point?.score)
      })
    })

    return Array.from(rowMap.values()).sort((a, b) => String(a.date).localeCompare(String(b.date)))
  }, [comparisonByMode])

  const summary = useMemo(() => {
    if (!chartData.length) return null
    const first = chartData[0]?.score ?? 0
    const last = chartData[chartData.length - 1]?.score ?? 0
    const trend = Math.abs(last - first) < 1 ? 'flat' : last > first ? 'rising' : 'falling'
    const peak = chartData.reduce((best, row) => (row.score > best.score ? row : best), chartData[0])
    const daysHighRisk = chartData.filter((row) => row.score >= 60).length
    return { trend, peak, daysHighRisk }
  }, [chartData])

  return (
    <Card className="p-4">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-[0.12em] text-text-primary">Anomaly Risk Forecast</h3>
          <p className="mt-1 text-xs text-text-secondary">
            {forecastData?.source_label || 'Hybrid anomaly projection with advanced + DL signals.'}
          </p>
        </div>
        <div className="flex items-center gap-1 rounded-md border border-card-border bg-surface/30 p-1">
          {DAY_OPTIONS.map((days) => (
            <button
              key={days}
              onClick={() => setSelectedDays(days)}
              className={clsx(
                'rounded px-2 py-1 text-[11px] font-semibold transition-colors',
                selectedDays === days
                  ? 'bg-brand-blue text-white'
                  : 'text-text-secondary hover:bg-surface hover:text-text-primary'
              )}
            >
              {days}D
            </button>
          ))}
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-1.5 rounded-md border border-card-border bg-surface/20 p-1.5">
        {MODE_OPTIONS.map((mode) => (
          <button
            key={mode.id}
            onClick={() => setSelectedMode(mode.id)}
            className={clsx(
              'rounded px-2.5 py-1 text-[11px] font-semibold transition-colors',
              selectedMode === mode.id
                ? 'bg-brand-blue text-white'
                : 'text-text-secondary hover:bg-surface hover:text-text-primary'
            )}
          >
            {mode.label}
          </button>
        ))}
      </div>

      <div className="mb-3 grid grid-cols-1 gap-2 text-xs">
        <div className="rounded-md border border-card-border bg-surface/30 px-2.5 py-2 text-text-secondary">
          Signal mode: <span className="font-semibold text-text-primary">{modeMeta?.label || selectedMode}</span>
        </div>
        <div className="rounded-md border border-card-border bg-surface/30 px-2.5 py-2 text-text-secondary">
          Models used: <span className="font-mono text-text-primary">{modelsUsed.length ? modelsUsed.join(', ') : 'N/A'}</span>
        </div>
      </div>

      {error && (
        <div className="mb-3 flex items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          <AlertTriangle size={14} />
          <span>{error}</span>
        </div>
      )}

      {loading && chartData.length > 0 && (
        <div className="mb-3 inline-flex items-center gap-2 rounded-md border border-card-border bg-surface/50 px-2 py-1 text-xs text-text-secondary">
          <Loader2 size={12} className="animate-spin" />
          Updating...
        </div>
      )}

      {!chartData.length && loading ? (
        <div className="h-[280px] animate-pulse rounded-lg bg-surface" />
      ) : !chartData.length ? (
        <div className="rounded-lg border border-dashed border-card-border bg-surface/40 px-4 py-6 text-center text-sm text-text-muted">
          No anomaly forecast available.
        </div>
      ) : (
        <>
          <div className="h-[280px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="riskBand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#1D6FDC" stopOpacity={0.24} />
                    <stop offset="100%" stopColor="#1D6FDC" stopOpacity={0.06} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#DBE4EF" strokeDasharray="4 4" vertical={false} />
                <XAxis dataKey="date" tickFormatter={(value) => formatDate(value)} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis domain={[0, 100]} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} width={34} />
                <ReferenceLine y={40} stroke="#F59E0B" strokeDasharray="3 3" />
                <ReferenceLine y={60} stroke="#EA580C" strokeDasharray="3 3" />
                <ReferenceLine y={75} stroke="#DC2626" strokeDasharray="3 3" />
                <Tooltip content={<RiskTooltip />} />
                <Area type="monotone" dataKey="upper" stroke="none" fill="url(#riskBand)" />
                <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" />
                <Line type="monotone" dataKey="score" stroke="#1D6FDC" strokeWidth={2.4} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-3 rounded-lg border border-card-border bg-surface/20 p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-[0.1em] text-text-primary">Mode Comparison</p>
              <span className="text-[11px] text-text-muted">Overlay of all anomaly forecast modes</span>
            </div>

            <div className="mb-2 flex flex-wrap items-center gap-3 text-[11px] text-text-secondary">
              {MODE_OPTIONS.map((mode) => (
                <span key={`legend-${mode.id}`} className="inline-flex items-center gap-1.5">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: MODE_LINE_COLORS[mode.id] }}
                  />
                  {mode.label}
                </span>
              ))}
            </div>

            {comparisonLoading && !comparisonChartData.length ? (
              <div className="h-[180px] animate-pulse rounded-md bg-surface" />
            ) : !comparisonChartData.length ? (
              <div className="rounded-md border border-dashed border-card-border bg-surface/40 px-3 py-5 text-center text-xs text-text-muted">
                Comparison chart unavailable for current selection.
              </div>
            ) : (
              <div className="h-[210px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparisonChartData} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid stroke="#DBE4EF" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(value) => formatDate(value)}
                      tick={{ fill: '#7C8BA1', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      domain={[0, 100]}
                      tick={{ fill: '#7C8BA1', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={34}
                    />
                    <Tooltip />

                    {availableComparisonModes.map((mode) => (
                      <Line
                        key={`compare-${mode.id}`}
                        type="monotone"
                        dataKey={mode.id}
                        stroke={MODE_LINE_COLORS[mode.id]}
                        strokeWidth={mode.id === selectedMode ? 2.4 : 1.8}
                        dot={false}
                        connectNulls
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {summary && (
            <div className="mt-3 grid grid-cols-1 gap-2 text-xs">
              <div className="rounded-md border border-card-border bg-surface/30 px-2.5 py-2 text-text-secondary">
                Peak day: <span className="font-mono text-text-primary">{formatDate(summary.peak.date, 'MMM dd')} ({formatScore(summary.peak.score)})</span>
              </div>
              <div className="rounded-md border border-card-border bg-surface/30 px-2.5 py-2 text-text-secondary">
                Trend: <span className="font-semibold capitalize text-text-primary">{summary.trend}</span>
              </div>
              <div className="rounded-md border border-card-border bg-surface/30 px-2.5 py-2 text-text-secondary">
                Days ≥ 60: <span className="font-mono text-text-primary">{summary.daysHighRisk}</span>
              </div>
            </div>
          )}

          <div className="mt-3 max-h-[180px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-card-border">
                  {['Date', 'Score', 'Lower', 'Upper', 'Risk'].map((heading) => (
                    <th key={heading} className="px-1.5 pb-1.5 text-left font-semibold uppercase tracking-[0.1em] text-text-secondary">
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {chartData.map((row) => (
                  <tr key={`${row.date}-${row.score}`} className="border-b border-card-border/50">
                    <td className="px-1.5 py-1.5 font-mono text-text-secondary">{formatDate(row.date, 'MMM dd, yyyy')}</td>
                    <td className="px-1.5 py-1.5 font-mono text-text-primary">{formatScore(row.score)}</td>
                    <td className="px-1.5 py-1.5 font-mono text-text-primary">{formatScore(row.lower)}</td>
                    <td className="px-1.5 py-1.5 font-mono text-text-primary">{formatScore(row.upper)}</td>
                    <td className="px-1.5 py-1.5">
                      <Badge variant={getRiskVariant(row.score)}>{row.risk_label || 'Unknown'}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  )
}
