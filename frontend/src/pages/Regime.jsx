/**
 * Regime.jsx — Phase 3 Addition
 * HMM Market Regime Timeline and Statistics
 */
import { useState, useEffect } from 'react'
import { Activity, TrendingUp, TrendingDown, AlertTriangle, BarChart3 } from 'lucide-react'
import { Card } from '../components/ui/Card'
import KPICard from '../components/cards/KPICard'
import { API_BASE } from '../constants/config'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import clsx from 'clsx'

const REGIME_COLORS = {
  bull: { bg: 'rgba(34, 197, 94, 0.15)', border: 'rgb(34, 197, 94)', text: 'text-green-500' },
  bear: { bg: 'rgba(251, 191, 36, 0.15)', border: 'rgb(251, 191, 36)', text: 'text-amber-500' },
  crisis: { bg: 'rgba(239, 68, 68, 0.15)', border: 'rgb(239, 68, 68)', text: 'text-red-500' },
}

const REGIME_ICONS = {
  bull: TrendingUp,
  bear: TrendingDown,
  crisis: AlertTriangle,
}

export default function Regime({ asset, loading: parentLoading }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!asset) return
    setLoading(true)
    fetch(`${API_BASE}/anomaly/regime/${asset}`)
      .then(res => res.json())
      .then(json => {
        if (json.error) {
          setError(json.error)
        } else {
          setData(json)
          setError(null)
        }
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [asset])

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Market Regime Analysis</h1>
          <p className="text-text-secondary text-sm">HMM-based market state detection</p>
        </div>
        <Card>
          <div className="text-center py-8 text-text-secondary">
            <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-amber-500" />
            <p>{error}</p>
            <p className="text-sm mt-2">Run Phase 2 training to enable regime detection.</p>
          </div>
        </Card>
      </div>
    )
  }

  const isLoading = loading || parentLoading
  const current = data?.current_regime ?? 'unknown'
  const stats = data?.regime_stats ?? {}
  const avgReturns = data?.avg_returns ?? {}
  const transitions = data?.transitions ?? {}
  const timeline = data?.timeline ?? []

  // Prepare chart data (last 252 trading days ~ 1 year)
  const chartData = timeline.slice(-252).map(d => ({
    date: d.date,
    score: d.score,
    regime: d.regime,
  }))

  // Calculate regime periods for color bands
  const regimePeriods = []
  let currentPeriod = null
  chartData.forEach((d, i) => {
    if (!currentPeriod || currentPeriod.regime !== d.regime) {
      if (currentPeriod) {
        currentPeriod.endIdx = i - 1
        regimePeriods.push(currentPeriod)
      }
      currentPeriod = { regime: d.regime, startIdx: i, endIdx: i }
    }
  })
  if (currentPeriod) {
    currentPeriod.endIdx = chartData.length - 1
    regimePeriods.push(currentPeriod)
  }

  const CurrentIcon = REGIME_ICONS[current] ?? Activity

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Market Regime Analysis</h1>
          <p className="text-text-secondary text-sm">HMM-based market state detection for {asset}</p>
        </div>
        {!isLoading && (
          <div className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium',
            current === 'bull' && 'bg-green-500/20 text-green-400',
            current === 'bear' && 'bg-amber-500/20 text-amber-400',
            current === 'crisis' && 'bg-red-500/20 text-red-400',
          )}>
            <CurrentIcon size={16} />
            <span className="uppercase">{current} Regime</span>
          </div>
        )}
      </div>

      {/* Regime Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {['bull', 'bear', 'crisis'].map((regime, idx) => {
          const stat = stats[regime] ?? { count: 0, pct: 0 }
          const avgRet = avgReturns[regime] ?? 0
          const Icon = REGIME_ICONS[regime]
          return (
            <Card key={regime}>
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-text-secondary text-sm uppercase tracking-wide mb-1">{regime} Regime</p>
                  <p className="font-mono text-2xl font-bold text-text-primary">{stat.pct?.toFixed(1)}%</p>
                  <p className="text-text-secondary text-sm mt-1">{stat.count?.toLocaleString()} trading days</p>
                  <p className={clsx(
                    'text-sm mt-2 font-mono',
                    avgRet >= 0 ? 'text-green-400' : 'text-red-400'
                  )}>
                    Avg: {avgRet >= 0 ? '+' : ''}{avgRet.toFixed(4)}%/day
                  </p>
                </div>
                <div className={clsx(
                  'p-3 rounded-lg',
                  regime === 'bull' && 'bg-green-500/20',
                  regime === 'bear' && 'bg-amber-500/20',
                  regime === 'crisis' && 'bg-red-500/20',
                )}>
                  <Icon className={clsx(
                    'w-6 h-6',
                    regime === 'bull' && 'text-green-400',
                    regime === 'bear' && 'text-amber-400',
                    regime === 'crisis' && 'text-red-400',
                  )} />
                </div>
              </div>
            </Card>
          )
        })}
      </div>

      {/* Regime Timeline Chart */}
      <Card>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Regime Timeline (1 Year)</h2>
          <p className="text-text-secondary text-sm">Anomaly score colored by market regime</p>
        </div>
        {isLoading ? (
          <div className="h-[300px] bg-surface rounded-lg animate-pulse" />
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="rgb(99, 102, 241)" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="rgb(99, 102, 241)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="date"
                tick={{ fill: '#6b7280', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(d) => d?.slice(5, 10)}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fill: '#6b7280', fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: 8,
                }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value, name) => [value.toFixed(1), 'Score']}
                labelFormatter={(label) => label}
              />
              <ReferenceLine y={60} stroke="#f59e0b" strokeDasharray="3 3" />
              <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="3 3" />
              <Area
                type="monotone"
                dataKey="score"
                stroke="rgb(99, 102, 241)"
                strokeWidth={2}
                fill="url(#scoreGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </Card>

      {/* Transition Matrix */}
      <Card>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Regime Transitions</h2>
          <p className="text-text-secondary text-sm">How often the market switches between states</p>
        </div>
        {isLoading ? (
          <div className="h-32 bg-surface rounded-lg animate-pulse" />
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(transitions).map(([key, count]) => {
              const [from, to] = key.split('->')
              return (
                <div key={key} className="bg-surface-alt p-3 rounded-lg">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={clsx(
                      'text-sm font-medium uppercase',
                      REGIME_COLORS[from]?.text ?? 'text-text-secondary'
                    )}>{from}</span>
                    <span className="text-text-secondary">→</span>
                    <span className={clsx(
                      'text-sm font-medium uppercase',
                      REGIME_COLORS[to]?.text ?? 'text-text-secondary'
                    )}>{to}</span>
                  </div>
                  <p className="font-mono text-lg font-bold text-text-primary">{count}</p>
                  <p className="text-text-secondary text-xs">transitions</p>
                </div>
              )
            })}
          </div>
        )}
      </Card>
    </div>
  )
}
