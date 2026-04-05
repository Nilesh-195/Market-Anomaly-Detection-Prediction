import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, TrendingDown, TrendingUp } from 'lucide-react'
import clsx from 'clsx'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card } from '../components/ui/Card'
import { API_BASE } from '../constants/config'
import { formatDate } from '../utils/formatters'

const REGIME_COLORS = {
  bull: { fill: 'rgba(16, 185, 129, 0.12)', text: 'text-emerald-700', pill: 'bg-emerald-50 border-emerald-200 text-emerald-700' },
  bear: { fill: 'rgba(245, 158, 11, 0.12)', text: 'text-amber-700', pill: 'bg-amber-50 border-amber-200 text-amber-700' },
  crisis: { fill: 'rgba(239, 68, 68, 0.14)', text: 'text-red-700', pill: 'bg-red-50 border-red-200 text-red-700' },
}

const REGIME_ICONS = {
  bull: TrendingUp,
  bear: TrendingDown,
  crisis: AlertTriangle,
}

function RegimeTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  return (
    <div className="min-w-[180px] rounded-xl border border-card-border bg-white p-3 shadow-float">
      <div className="text-xs font-semibold uppercase tracking-[0.12em] text-text-muted">
        {formatDate(label, 'MMM dd, yyyy')}
      </div>
      <div className="mt-2 flex items-center justify-between text-sm">
        <span className="text-text-secondary">Score</span>
        <span className="font-mono font-semibold text-brand-blue">{row?.score?.toFixed(1)}</span>
      </div>
      <div className="mt-1 text-xs font-semibold uppercase tracking-[0.08em] text-text-secondary">
        Regime: {row?.regime}
      </div>
    </div>
  )
}

export default function Regime({ asset, loading: parentLoading }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!asset) return

    fetch(`${API_BASE}/anomaly/regime/${asset}`)
      .then((res) => res.json())
      .then((json) => {
        if (json.error) {
          setError(json.error)
        } else {
          setData(json)
          setError(null)
        }
      })
      .catch((err) => setError(err.message))
  }, [asset])

  const isLoading = !data || parentLoading
  const current = data?.current_regime ?? 'unknown'
  const stats = data?.regime_stats ?? {}
  const avgReturns = data?.avg_returns ?? {}
  const transitions = data?.transitions ?? {}
  const timeline = (data?.timeline ?? []).slice(-252)

  const regimePeriods = useMemo(() => {
    const periods = []
    let currentPeriod = null

    timeline.forEach((row, index) => {
      if (!currentPeriod || currentPeriod.regime !== row.regime) {
        if (currentPeriod) {
          currentPeriod.end = timeline[index - 1]?.date
          periods.push(currentPeriod)
        }
        currentPeriod = { regime: row.regime, start: row.date, end: row.date }
      }
    })

    if (currentPeriod) {
      currentPeriod.end = timeline[timeline.length - 1]?.date
      periods.push(currentPeriod)
    }

    return periods
  }, [timeline])

  const scoreValues = timeline.map((row) => row.score).filter(Number.isFinite)
  const yDomain = scoreValues.length
    ? [Math.min(...scoreValues) * 0.94, Math.max(...scoreValues) * 1.06]
    : ['auto', 'auto']

  const CurrentIcon = REGIME_ICONS[current] ?? Activity

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Market Regime Identification</h1>
          <p className="text-sm text-text-secondary">HMM-driven market state timeline.</p>
        </div>
        <Card>
          <div className="py-8 text-center text-text-secondary">
            <AlertTriangle className="mx-auto mb-3 h-12 w-12 text-amber-500" />
            <p>{error}</p>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Market Regime Identification</h1>
          <p className="text-sm text-text-secondary">HMM market state timeline for {asset}.</p>
        </div>
        {!isLoading && (
          <div className={clsx('inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-semibold uppercase', REGIME_COLORS[current]?.pill)}>
            <CurrentIcon size={16} />
            {current}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {['bull', 'bear', 'crisis'].map((regime) => {
          const stat = stats[regime] ?? { count: 0, pct: 0 }
          const avgRet = avgReturns[regime] ?? 0
          const Icon = REGIME_ICONS[regime]

          return (
            <Card key={regime}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.14em] text-text-muted">{regime} regime</p>
                  <p className="font-mono text-2xl font-bold text-text-primary">{stat.pct?.toFixed(1)}%</p>
                  <p className="text-xs text-text-secondary">{stat.count?.toLocaleString()} trading days</p>
                  <p className={clsx('mt-2 text-xs font-mono', avgRet >= 0 ? 'text-emerald-700' : 'text-red-700')}>
                    Avg {avgRet >= 0 ? '+' : ''}{avgRet.toFixed(4)}% / day
                  </p>
                </div>
                <div className={clsx('rounded-lg p-2', REGIME_COLORS[regime]?.pill)}>
                  <Icon size={18} />
                </div>
              </div>
            </Card>
          )
        })}
      </div>

      <Card>
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Regime Timeline</h2>
        <p className="mb-4 text-sm text-text-secondary">Background colors map bull, bear, and crisis state transitions.</p>
        {isLoading ? (
          <div className="h-[320px] animate-pulse rounded-lg bg-surface" />
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <AreaChart data={timeline} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
              <defs>
                <linearGradient id="regimeScoreGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#1D6FDC" stopOpacity={0.32} />
                  <stop offset="100%" stopColor="#1D6FDC" stopOpacity={0.04} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#DBE4EF" strokeDasharray="4 4" vertical={false} />
              {regimePeriods.map((period) => (
                <ReferenceArea
                  key={`${period.regime}-${period.start}`}
                  x1={period.start}
                  x2={period.end}
                  y1={yDomain[0]}
                  y2={yDomain[1]}
                  fill={REGIME_COLORS[period.regime]?.fill}
                  fillOpacity={1}
                  ifOverflow="extendDomain"
                />
              ))}
              <XAxis dataKey="date" tickFormatter={(value) => formatDate(value)} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis domain={yDomain} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} width={48} />
              <Tooltip content={<RegimeTooltip />} />
              <ReferenceLine y={60} stroke="#f59e0b" strokeDasharray="3 3" />
              <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="3 3" />
              <Area type="monotone" dataKey="score" stroke="#0B3A63" strokeWidth={2.3} fill="url(#regimeScoreGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </Card>

      <Card>
        <h2 className="mb-3 text-lg font-semibold text-text-primary">Regime Transition Activity</h2>
        <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
          {Object.entries(transitions).map(([key, count]) => {
            const [from, to] = key.split('->')
            return (
              <div key={key} className="rounded-lg border border-card-border bg-surface/60 p-3">
                <div className="text-xs uppercase tracking-[0.1em] text-text-muted">{from} to {to}</div>
                <div className="mt-1 font-mono text-2xl font-bold text-text-primary">{count}</div>
              </div>
            )
          })}
        </div>
      </Card>
    </div>
  )
}
