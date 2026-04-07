import { useEffect, useMemo, useState } from 'react'
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
import clsx from 'clsx'
import { AlertTriangle, Loader2 } from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { fetchAnomalyForecast } from '../../services/api'
import { formatDate, formatScore } from '../../utils/formatters'

const DAY_OPTIONS = [5, 10, 15, 30]

function getRiskVariant(score) {
  if (score >= 75) return 'red'
  if (score >= 60) return 'orange'
  if (score >= 40) return 'yellow'
  return 'green'
}

function isMatchingForecast(data, asset, days) {
  return data?.asset === asset && Number(data?.horizon) === Number(days) && Array.isArray(data?.forecast)
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
  const [selectedDays, setSelectedDays] = useState(defaultDays)
  const [forecastData, setForecastData] = useState(
    isMatchingForecast(initialForecast, asset, defaultDays) ? initialForecast : null
  )
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!asset) return

    if (isMatchingForecast(initialForecast, asset, selectedDays)) {
      setForecastData(initialForecast)
      setError(null)
      return
    }

    let active = true
    setLoading(true)
    setError(null)

    fetchAnomalyForecast(asset, selectedDays)
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
  }, [asset, selectedDays, initialForecast])

  const rows = Array.isArray(forecastData?.forecast) ? forecastData.forecast : []
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
          <p className="mt-1 text-xs text-text-secondary">ARIMA projection of anomaly risk score with confidence interval.</p>
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
        <div className="h-[220px] animate-pulse rounded-lg bg-surface" />
      ) : !chartData.length ? (
        <div className="rounded-lg border border-dashed border-card-border bg-surface/40 px-4 py-6 text-center text-sm text-text-muted">
          No anomaly forecast available.
        </div>
      ) : (
        <>
          <div className="h-[220px] w-full">
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

          {summary && (
            <div className="mt-3 grid grid-cols-1 gap-2 text-xs md:grid-cols-3">
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
