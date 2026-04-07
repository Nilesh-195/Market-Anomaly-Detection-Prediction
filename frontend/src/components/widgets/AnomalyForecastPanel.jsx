import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { AlertTriangle, Loader2, Minus, TrendingDown, TrendingUp } from 'lucide-react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { COLOURS } from '../../constants/colours'
import { fetchAnomalyForecast } from '../../services/api'
import { formatDate, formatScore } from '../../utils/formatters'
import { getRiskColor, getRiskLabel } from '../../utils/riskHelpers'

const HORIZON_OPTIONS = [5, 10, 15, 30]
const MODE_OPTIONS = [
  { id: 'hybrid', label: 'Hybrid' },
  { id: 'dl', label: 'DL Composite' },
  { id: 'advanced', label: 'Advanced' },
  { id: 'ensemble', label: 'Baseline' },
]

function getBadgeVariant(score) {
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

function toDateSafe(value) {
  if (!value) return null
  const d = new Date(value)
  return Number.isNaN(d.getTime()) ? null : d
}

function ForecastTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  const score = Number(row?.score ?? row?.observedScore)
  const isObserved = row?.kind === 'observed'

  return (
    <div className="min-w-[220px] rounded-xl border border-card-border bg-card-bg p-3 shadow-lg">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-text-secondary">
        {formatDate(label, 'MMM dd, yyyy')}
      </div>
      <div className="space-y-1.5 text-sm">
        <div className="text-xs text-text-muted">{isObserved ? 'Observed (actual)' : 'Forecast (model-driven)'}</div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-text-secondary">Score</span>
          <span className="font-mono font-bold" style={{ color: getRiskColor(score) }}>{formatScore(score)}</span>
        </div>
        {!isObserved && (
          <>
            <div className="flex items-center justify-between gap-4">
              <span className="text-text-secondary">Lower</span>
              <span className="font-mono font-semibold text-text-primary">{formatScore(row?.lower)}</span>
            </div>
            <div className="flex items-center justify-between gap-4">
              <span className="text-text-secondary">Upper</span>
              <span className="font-mono font-semibold text-text-primary">{formatScore(row?.upper)}</span>
            </div>
          </>
        )}
        <div className="border-t border-card-border pt-1.5 text-xs text-text-muted">
          Risk: {row?.risk_label || getRiskLabel(score)}
        </div>
      </div>
    </div>
  )
}

export default function AnomalyForecastPanel({ asset, initialForecast, latestObservation = null, defaultDays = 10 }) {
  const defaultMode = String(initialForecast?.mode || 'hybrid').toLowerCase()
  const [selectedDays, setSelectedDays] = useState(defaultDays)
  const [selectedMode, setSelectedMode] = useState(defaultMode)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [forecastData, setForecastData] = useState(
    isMatchingForecast(initialForecast, asset, defaultDays, defaultMode) ? initialForecast : null
  )

  useEffect(() => {
    if (!asset) return

    if (isMatchingForecast(initialForecast, asset, selectedDays, selectedMode)) {
      setForecastData(initialForecast)
      setError(null)
      setLoading(false)
      return
    }

    let active = true
    setLoading(true)
    setError(null)

    fetchAnomalyForecast(asset, selectedDays, { mode: selectedMode, method: 'hybrid' })
      .then((data) => {
        if (!active) return
        if (!Array.isArray(data?.forecast)) {
          throw new Error('Forecast response is missing expected data')
        }
        setForecastData(data)
      })
      .catch((err) => {
        if (!active) return
        setError(err?.message || 'Failed to load anomaly forecast')
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => {
      active = false
    }
  }, [asset, initialForecast, selectedDays, selectedMode])

  const forecastPoints = Array.isArray(forecastData?.forecast) ? forecastData.forecast : []
  const modeMeta = MODE_OPTIONS.find((item) => item.id === selectedMode)
  const modelsUsed = Array.isArray(forecastData?.models_used) ? forecastData.models_used : []

  const chartData = useMemo(
    () => forecastPoints.map((point) => {
      const score = Number(point?.score) || 0
      const lower = Number(point?.lower)
      const upper = Number(point?.upper)

      return {
        date: point?.date,
        kind: 'forecast',
        observedScore: null,
        score,
        lower: Number.isFinite(lower) ? lower : score,
        upper: Number.isFinite(upper) ? upper : score,
        risk_label: point?.risk_label || getRiskLabel(score),
      }
    }),
    [forecastPoints]
  )

  const observedPoint = useMemo(() => {
    const score = Number(latestObservation?.score)
    if (!latestObservation?.date || !Number.isFinite(score)) return null

    return {
      date: latestObservation.date,
      kind: 'observed',
      observedScore: score,
      score: null,
      lower: null,
      upper: null,
      risk_label: getRiskLabel(score),
    }
  }, [latestObservation])

  const displayData = useMemo(() => {
    if (!observedPoint) return chartData
    return [observedPoint, ...chartData]
  }, [chartData, observedPoint])

  const firstForecast = chartData[0] ?? null

  const generatedDate = forecastData?.generated || null
  const generatedAt = toDateSafe(generatedDate)
  const observedAt = toDateSafe(observedPoint?.date)
  const staleDays = generatedAt && observedAt
    ? Math.abs(Math.round((generatedAt - observedAt) / (1000 * 60 * 60 * 24)))
    : null
  const isSnapshotStale = staleDays != null && staleDays > 3

  const summary = useMemo(() => {
    if (!chartData.length) return null

    const peak = chartData.reduce((maxRow, row) => (row.score > maxRow.score ? row : maxRow), chartData[0])
    const first = chartData[0].score
    const last = chartData[chartData.length - 1].score
    const delta = last - first
    const trend = Math.abs(delta) < 1 ? 'flat' : delta > 0 ? 'rising' : 'falling'
    const above60 = chartData.filter((row) => row.score >= 60).length

    return {
      peak,
      trend,
      above60,
      delta,
    }
  }, [chartData])

  const trendIcon = summary?.trend === 'rising'
    ? <TrendingUp size={14} className="text-red-600" />
    : summary?.trend === 'falling'
      ? <TrendingDown size={14} className="text-emerald-600" />
      : <Minus size={14} className="text-text-muted" />

  return (
    <Card>
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-lg font-bold text-text-primary">Anomaly Prediction (Next {selectedDays} Days)</h2>
          <p className="mt-1 text-xs text-text-secondary">
            {forecastData?.source_label || 'Hybrid forecast using advanced + deep-learning anomaly signals.'}
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2 rounded-lg border border-card-border bg-surface/40 p-1">
            {HORIZON_OPTIONS.map((days) => (
              <button
                key={days}
                onClick={() => setSelectedDays(days)}
                className={clsx(
                  'rounded-md px-2.5 py-1.5 text-xs font-semibold transition-colors',
                  selectedDays === days
                    ? 'bg-brand-blue text-white shadow-sm'
                    : 'text-text-secondary hover:bg-surface hover:text-text-primary'
                )}
              >
                {days}D
              </button>
            ))}
          </div>

          <div className="flex items-center gap-1 rounded-lg border border-card-border bg-surface/40 p-1">
            {MODE_OPTIONS.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setSelectedMode(mode.id)}
                className={clsx(
                  'rounded-md px-2 py-1 text-[11px] font-semibold transition-colors',
                  selectedMode === mode.id
                    ? 'bg-brand-blue text-white shadow-sm'
                    : 'text-text-secondary hover:bg-surface hover:text-text-primary'
                )}
              >
                {mode.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="mb-4 grid grid-cols-1 gap-2 text-xs md:grid-cols-3">
        <div className="rounded-md border border-card-border bg-surface/30 px-3 py-2 text-text-secondary">
          Signal mode: <span className="font-semibold text-text-primary">{modeMeta?.label || selectedMode}</span>
        </div>
        <div className="rounded-md border border-card-border bg-surface/30 px-3 py-2 text-text-secondary">
          Models used: <span className="font-mono text-text-primary">{modelsUsed.length ? modelsUsed.join(', ') : 'Unavailable'}</span>
        </div>
        <div className="rounded-md border border-card-border bg-surface/30 px-3 py-2 text-text-secondary">
          Forecast generated: <span className="font-mono text-text-primary">{generatedDate ? formatDate(generatedDate, 'MMM dd, yyyy') : 'Unavailable'}</span>
        </div>
      </div>

      {isSnapshotStale && (
        <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Forecast is real, but based on the latest available model snapshot ({staleDays} day gap from latest observed point).
        </div>
      )}

      {error && (
        <div className="mb-4 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {loading && chartData.length > 0 && (
        <div className="mb-3 inline-flex items-center gap-2 rounded-md border border-card-border bg-surface/60 px-2 py-1 text-xs text-text-secondary">
          <Loader2 size={12} className="animate-spin" />
          Updating forecast...
        </div>
      )}

      {!chartData.length && loading ? (
        <div className="space-y-4">
          <div className="h-[280px] animate-pulse rounded-lg bg-surface" />
          <div className="h-32 animate-pulse rounded-lg bg-surface" />
        </div>
      ) : !chartData.length ? (
        <div className="h-[220px] rounded-lg border border-dashed border-card-border bg-surface/50 px-4 py-6 text-center text-sm text-text-muted">
          No forecast data available for this asset and horizon.
        </div>
      ) : (
        <>
          <div className="mb-5 h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={displayData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="anomalyRiskBand" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={COLOURS.chartBlue} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={COLOURS.chartBlue} stopOpacity={0.06} />
                  </linearGradient>
                </defs>

                <CartesianGrid stroke={COLOURS.cardBorder} strokeDasharray="3 3" vertical={false} opacity={0.7} />
                <XAxis
                  dataKey="date"
                  tickFormatter={(value) => formatDate(value, 'MMM dd')}
                  axisLine={false}
                  tickLine={false}
                  minTickGap={20}
                  tick={{ fill: COLOURS.textMuted, fontSize: 11, fontFamily: 'monospace' }}
                />
                <YAxis
                  domain={[0, 100]}
                  axisLine={false}
                  tickLine={false}
                  width={34}
                  orientation="right"
                  tick={{ fill: COLOURS.textMuted, fontSize: 11, fontFamily: 'monospace' }}
                />

                <ReferenceLine y={40} stroke={COLOURS.riskElevated} strokeDasharray="4 4" opacity={0.7} label={{ value: 'Elevated', position: 'insideTopLeft', fill: COLOURS.riskElevated, fontSize: 10 }} />
                <ReferenceLine y={60} stroke={COLOURS.riskHigh} strokeDasharray="4 4" opacity={0.7} label={{ value: 'High Risk', position: 'insideTopLeft', fill: COLOURS.riskHigh, fontSize: 10 }} />
                <ReferenceLine y={75} stroke={COLOURS.riskExtreme} strokeDasharray="4 4" opacity={0.7} label={{ value: 'Extreme', position: 'insideTopLeft', fill: COLOURS.riskExtreme, fontSize: 10 }} />

                <Tooltip content={<ForecastTooltip />} cursor={{ stroke: COLOURS.textMuted, strokeDasharray: '4 4', strokeWidth: 1 }} />

                <Area type="monotone" dataKey="upper" stroke="none" fill="url(#anomalyRiskBand)" connectNulls />
                <Area type="monotone" dataKey="lower" stroke="none" fill={COLOURS.cardBg} connectNulls />

                <Line
                  type="monotone"
                  dataKey="observedScore"
                  stroke={COLOURS.brandBlue}
                  strokeWidth={2.2}
                  dot={{ r: 4, fill: COLOURS.brandBlue, stroke: '#fff', strokeWidth: 1.5 }}
                  connectNulls={false}
                  isAnimationActive={false}
                />

                <Line
                  type="monotone"
                  dataKey="score"
                  stroke={COLOURS.chartBlue}
                  strokeWidth={2.4}
                  strokeDasharray="6 4"
                  dot={false}
                  activeDot={{ r: 4, fill: COLOURS.chartBlue, stroke: '#fff', strokeWidth: 2 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {observedPoint && firstForecast && (
            <div className="mb-4 rounded-lg border border-brand-blue/20 bg-brand-blue/5 px-3 py-2 text-sm text-text-primary">
              Previous observed score: <span className="font-mono font-semibold">{formatScore(observedPoint.observedScore)}</span> on <span className="font-mono">{formatDate(observedPoint.date, 'MMM dd, yyyy')}</span>
              {' '}→ First predicted score: <span className="font-mono font-semibold">{formatScore(firstForecast.score)}</span> on <span className="font-mono">{formatDate(firstForecast.date, 'MMM dd, yyyy')}</span>
            </div>
          )}

          {summary && (
            <div className="mb-4 grid grid-cols-1 gap-3 md:grid-cols-3">
              <div className="rounded-lg border border-card-border bg-surface/40 p-3">
                <div className="text-[11px] uppercase tracking-wider text-text-muted">Peak Risk</div>
                <div className="mt-1 flex items-baseline gap-2">
                  <span className="font-mono text-xl font-bold" style={{ color: getRiskColor(summary.peak.score) }}>
                    {formatScore(summary.peak.score)}
                  </span>
                  <span className="text-xs text-text-secondary">on {formatDate(summary.peak.date, 'MMM dd')}</span>
                </div>
              </div>

              <div className="rounded-lg border border-card-border bg-surface/40 p-3">
                <div className="text-[11px] uppercase tracking-wider text-text-muted">Trend</div>
                <div className="mt-1 flex items-center gap-2 text-sm font-semibold capitalize text-text-primary">
                  {trendIcon}
                  {summary.trend}
                </div>
                <div className="mt-1 font-mono text-xs text-text-secondary">Delta: {formatScore(summary.delta)}</div>
              </div>

              <div className="rounded-lg border border-card-border bg-surface/40 p-3">
                <div className="text-[11px] uppercase tracking-wider text-text-muted">Days Above 60</div>
                <div className="mt-1 font-mono text-xl font-bold text-text-primary">{summary.above60}</div>
                <div className="mt-1 text-xs text-text-secondary">High Risk threshold crossings</div>
              </div>
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-card-border">
                  {['Type', 'Date', 'Score', 'Lower', 'Upper', 'Risk Label'].map((heading) => (
                    <th key={heading} className="px-2 pb-2 text-left text-[11px] font-medium uppercase tracking-wider text-text-secondary">
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayData.map((point) => (
                  <tr key={`${point.kind}-${point.date}-${point.score ?? point.observedScore}`} className="border-b border-card-border/50 hover:bg-surface/40">
                    <td className="px-2 py-2.5">
                      <Badge variant={point.kind === 'observed' ? 'blue' : 'default'}>
                        {point.kind === 'observed' ? 'Observed' : 'Forecast'}
                      </Badge>
                    </td>
                    <td className="px-2 py-2.5 font-mono text-xs text-text-secondary">{formatDate(point.date, 'MMM dd, yyyy')}</td>
                    <td className="px-2 py-2.5 font-mono font-semibold" style={{ color: getRiskColor(point.score ?? point.observedScore) }}>{formatScore(point.score ?? point.observedScore)}</td>
                    <td className="px-2 py-2.5 font-mono text-text-primary">{point.kind === 'observed' ? '—' : formatScore(point.lower)}</td>
                    <td className="px-2 py-2.5 font-mono text-text-primary">{point.kind === 'observed' ? '—' : formatScore(point.upper)}</td>
                    <td className="px-2 py-2.5">
                      <Badge variant={getBadgeVariant(point.score ?? point.observedScore)}>{point.risk_label}</Badge>
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