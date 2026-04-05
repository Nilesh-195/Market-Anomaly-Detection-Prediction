import { useEffect, useMemo, useState } from 'react'
import { Activity, Layers, Sparkles, Zap } from 'lucide-react'
import clsx from 'clsx'
import { Card } from '../components/ui/Card'
import ModelConsensus from '../components/widgets/ModelConsensus'
import AnomalyTable from '../components/widgets/AnomalyTable'
import PriceAreaChart from '../components/charts/PriceAreaChart'
import RiskScoreChart from '../components/charts/RiskScoreChart'
import { formatPct, formatPrice, formatScore } from '../utils/formatters'
import { getRiskColor } from '../utils/riskHelpers'
import { API_BASE, ASSETS } from '../constants/config'

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

function Sparkline({ values = [] }) {
  if (!values.length) {
    return <div className="h-12 rounded-lg bg-surface" />
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

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-12 w-full overflow-visible">
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#1D6FDC" stopOpacity="0.36" />
          <stop offset="100%" stopColor="#1D6FDC" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <polyline
        points={points}
        fill="none"
        stroke="#1D6FDC"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <polygon points={`0,100 ${points} 100,100`} fill="url(#sparkFill)" />
    </svg>
  )
}

export default function Dashboard({ current, historical, priceForecast, loading, selectedAsset }) {
  const [advancedData, setAdvancedData] = useState(null)
  const [overviewCards, setOverviewCards] = useState([])
  const [overviewLoading, setOverviewLoading] = useState(true)

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
        const summaryRes = await fetch(`${API_BASE}/summary`)
        const summaryJson = await summaryRes.json()
        const summaryAssets = Array.isArray(summaryJson?.assets) ? summaryJson.assets : []

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
            riskLabel: item?.risk_label ?? 'Unknown',
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
  }, [selectedAsset])

  const score = current?.ensemble_score ?? 0
  const advScore = advancedData?.advanced_ensemble ?? score
  const regime = advancedData?.current_regime ?? 'unknown'

  const currentPrice = priceForecast?.current_price ?? current?.price ?? 0
  const tomorrowPrice = priceForecast?.forecast?.values?.[0] ?? currentPrice
  const tomorrowChange = currentPrice > 0 ? ((tomorrowPrice / currentPrice) - 1) * 100 : 0

  const chartData = historical?.chart_data ?? []
  const anomalyPts = useMemo(
    () => (historical?.events ?? []).slice(0, 40).map((event) => ({ date: event.date, close: null })),
    [historical]
  )

  return (
    <div className="space-y-6">
      <Card className="relative overflow-hidden border-brand-blue/15 bg-gradient-to-br from-white via-white to-sky-50/55">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_88%_12%,rgba(21,158,192,0.15),transparent_35%)]" />
        <div className="relative flex items-start justify-between gap-4">
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight text-text-primary">Global Overview Dashboard</h1>
            <p className="mt-2 text-sm text-text-secondary">
              Cross-asset command center with live pricing, forecast deltas, and anomaly risk signals.
            </p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-xl border border-card-border bg-white/80 px-3 py-2 text-xs font-mono text-text-secondary">
            <Sparkles size={14} className="text-brand-blue-dim" />
            {selectedAsset} focus
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
        {(overviewLoading ? Array.from({ length: 6 }) : overviewCards).map((card, index) => (
          <Card
            key={card?.asset ?? `skeleton-${index}`}
            hover
            className="overflow-hidden border-card-border/90 bg-gradient-to-br from-white to-surface/70"
          >
            {overviewLoading || !card ? (
              <div className="animate-pulse space-y-3">
                <div className="h-4 w-24 rounded bg-surface" />
                <div className="h-8 w-32 rounded bg-surface" />
                <div className="h-12 rounded bg-surface" />
              </div>
            ) : (
              <>
                <div className="mb-3 flex items-start justify-between gap-2">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">{card.asset}</div>
                    <div className="mt-1 text-sm font-semibold text-text-primary">{card.name}</div>
                  </div>
                  <span className={clsx('rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase', riskBadgeClasses(card.score))}>
                    {card.riskLabel}
                  </span>
                </div>

                <div className="mb-2 flex items-end justify-between gap-3">
                  <div className="font-mono text-2xl font-bold text-text-primary">{formatPrice(card.currentPrice)}</div>
                  <div className={clsx(
                    'text-xs font-mono font-semibold',
                    card.delta30dPct >= 0 ? 'text-emerald-600' : 'text-red-600'
                  )}>
                    30D {card.delta30dPct >= 0 ? '+' : ''}{formatPct(card.delta30dPct)}
                  </div>
                </div>

                <Sparkline values={card.sparkline} />
              </>
            )}
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <div className="mb-4 flex items-center gap-2">
            <Layers className="h-5 w-5 text-brand-blue-dim" />
            <h2 className="text-lg font-semibold text-text-primary">Selected Asset Context</h2>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.14em] text-text-muted">Baseline Score</p>
              <p className="font-mono text-3xl font-bold" style={{ color: getRiskColor(score) }}>
                {formatScore(score)}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.14em] text-text-muted">Advanced Score</p>
              <p className="font-mono text-3xl font-bold" style={{ color: getRiskColor(advScore) }}>
                {formatScore(advScore)}
              </p>
            </div>
          </div>
          <p className="mt-4 text-sm text-text-secondary">
            Current regime: <span className="font-semibold uppercase text-text-primary">{regime}</span>
          </p>
        </Card>

        <Card>
          <div className="mb-2 flex items-center gap-2">
            <Zap className="h-4 w-4 text-amber-500" />
            <h3 className="text-sm font-semibold uppercase tracking-[0.14em] text-text-primary">Near-term forecast</h3>
          </div>
          <div className="mt-3 flex items-end justify-between">
            <div>
              <p className="text-xs text-text-muted">Tomorrow target</p>
              <p className="font-mono text-3xl font-bold text-text-primary">{formatPrice(tomorrowPrice)}</p>
            </div>
            <p className={clsx('font-mono text-sm font-semibold', tomorrowChange >= 0 ? 'text-emerald-600' : 'text-red-600')}>
              {tomorrowChange >= 0 ? '+' : ''}{formatPct(tomorrowChange)}
            </p>
          </div>
          <div className="mt-4">
            <ModelConsensus current={current} />
          </div>
        </Card>
      </div>

      <Card>
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Anomaly Overlay Chart</h2>
        <p className="mb-4 text-sm text-text-secondary">Price action with highlighted anomaly zones for {selectedAsset}.</p>
        {loading ? <div className="h-[280px] animate-pulse rounded-lg bg-surface" /> : <PriceAreaChart data={chartData} anomalyPoints={anomalyPts} />}
      </Card>

      <Card>
        <h2 className="mb-1 text-lg font-semibold text-text-primary">Risk Timeline</h2>
        <p className="mb-4 text-sm text-text-secondary">Dynamic anomaly score trajectory with threshold awareness.</p>
        {loading ? <div className="h-[220px] animate-pulse rounded-lg bg-surface" /> : <RiskScoreChart data={chartData} />}
      </Card>

      <Card>
        <div className="mb-4 flex items-center gap-2">
          <Activity className="h-4 w-4 text-brand-blue-dim" />
          <h2 className="text-lg font-semibold text-text-primary">Recent Anomaly Events</h2>
        </div>
        <AnomalyTable events={historical?.events ?? []} loading={loading} maxRows={8} />
      </Card>
    </div>
  )
}
