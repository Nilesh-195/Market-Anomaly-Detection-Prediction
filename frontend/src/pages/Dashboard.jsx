import { useState, useEffect } from 'react'
import {
  DollarSign, Activity, Waves, AlertCircle, TrendingUp, TrendingDown, Layers, Zap,
} from 'lucide-react'
import { Card } from '../components/ui/Card'
import KPICard from '../components/cards/KPICard'
import ModelConsensus from '../components/widgets/ModelConsensus'
import AnomalyTable from '../components/widgets/AnomalyTable'
import PriceAreaChart from '../components/charts/PriceAreaChart'
import RiskScoreChart from '../components/charts/RiskScoreChart'
import {
  formatPrice, formatScore, formatZScore, formatVolatility, formatPct,
} from '../utils/formatters'
import { getRiskColor, getZScoreColor, getVolatilityColor } from '../utils/riskHelpers'
import { useLiveScores } from '../hooks/useWebSocket'
import { API_BASE } from '../constants/config'
import clsx from 'clsx'

const REGIME_COLORS = {
  bull: 'bg-green-500/20 text-green-400',
  bear: 'bg-amber-500/20 text-amber-400',
  crisis: 'bg-red-500/20 text-red-400',
}

export default function Dashboard({ current, historical, priceForecast, loading, selectedAsset }) {
  const [advancedData, setAdvancedData] = useState(null)
  const { data: liveData, connected } = useLiveScores(selectedAsset ?? 'SP500')

  // Fetch advanced analysis for regime and model tiers
  useEffect(() => {
    if (!selectedAsset) return
    fetch(`${API_BASE}/anomaly/advanced/${selectedAsset}`)
      .then(res => res.json())
      .then(setAdvancedData)
      .catch(console.error)
  }, [selectedAsset])

  const score     = current?.ensemble_score ?? 0
  const advScore  = advancedData?.advanced_ensemble ?? score
  const regime    = advancedData?.current_regime ?? liveData?.regime ?? 'unknown'
  const price     = current?.price
  const zscore    = current?.zscore
  const vol       = current?.volatility
  const anomCount = historical?.total_anomaly_days ?? 0

  // Price forecast data
  const currentPrice    = priceForecast?.current_price ?? price ?? 0
  const forecastValues  = priceForecast?.forecast?.values ?? []
  const tomorrowPrice   = forecastValues[0] ?? currentPrice
  const tomorrowChange  = currentPrice ? ((tomorrowPrice / currentPrice) - 1) * 100 : 0

  // chart_data comes from historical_anomalies endpoint
  const chartData  = historical?.chart_data ?? []
  const anomalyPts = (historical?.events ?? []).slice(0, 30).map(a => ({
    date:  a.date,
    close: null,
  }))

  return (
    <div className="space-y-6">
      {/* Header with Live Badge and Regime Pill */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-text-primary">Market Dashboard</h1>
            {connected && (
              <span className="flex items-center gap-1.5 px-2 py-0.5 bg-green-500/20 text-green-400 text-xs font-medium rounded-full">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                LIVE
              </span>
            )}
          </div>
          <p className="text-text-secondary text-sm">Real-time market anomaly detection and analysis</p>
        </div>
        {regime !== 'unknown' && (
          <div className={clsx(
            'flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium',
            REGIME_COLORS[regime] ?? 'bg-gray-500/20 text-gray-400'
          )}>
            <Layers size={16} />
            <span className="uppercase">{regime} Regime</span>
          </div>
        )}
      </div>

      {/* KPI Cards - Essential Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <KPICard
          label="Current Price" index={0} loading={loading}
          value={price ? formatPrice(price) : '—'}
          icon={DollarSign}
          delta={current?.price_change_pct ? `${current.price_change_pct > 0 ? '+' : ''}${current.price_change_pct?.toFixed(2)}%` : null}
        />
        <KPICard
          label="Risk Score" index={1} loading={loading}
          value={formatScore(score)}
          valueColor={getRiskColor(score)}
          icon={Activity}
          delta={current?.score_delta != null ? `${current.score_delta > 0 ? '+' : ''}${formatScore(current.score_delta)}` : null}
        />
        <KPICard
          label="Z-Score" index={2} loading={loading}
          value={zscore != null ? formatZScore(zscore) : '—'}
          valueColor={zscore != null ? getZScoreColor(zscore) : undefined}
          icon={() => <span className="text-text-primary text-sm">σ</span>}
        />
        <KPICard
          label="Volatility (30d)" index={3} loading={loading}
          value={vol != null ? formatVolatility(vol) : '—'}
          valueColor={vol != null ? getVolatilityColor(vol) : undefined}
          icon={Waves}
        />
        <KPICard
          label="Anomalies" index={4} loading={loading}
          value={String(anomCount)}
          icon={AlertCircle}
          delta={null}
          deltaLabel="detected"
        />
      </div>

      {/* Model Tier Comparison Card */}
      {advancedData && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Zap className="w-5 h-5 text-amber-400" />
            <h2 className="text-lg font-semibold text-text-primary">Model Tier Comparison</h2>
          </div>
          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-text-secondary text-sm mb-1">Baseline (4 models)</p>
              <p className="font-mono text-3xl font-bold" style={{ color: getRiskColor(score) }}>
                {formatScore(score)}
              </p>
              <p className="text-text-secondary text-xs mt-1">Z-Score, IForest, LSTM, Prophet</p>
            </div>
            <div>
              <p className="text-text-secondary text-sm mb-1">Advanced (7 models)</p>
              <p className="font-mono text-3xl font-bold" style={{ color: getRiskColor(advScore) }}>
                {formatScore(advScore)}
              </p>
              <p className="text-text-secondary text-xs mt-1">+ XGBoost, HMM, TCN</p>
            </div>
          </div>
        </Card>
      )}

      {/* Price Chart Section */}
      <Card>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Price History</h2>
          <p className="text-text-secondary text-sm">Historical price with anomaly markers</p>
        </div>
        {loading
          ? <div className="h-[300px] bg-surface rounded-lg animate-pulse" />
          : <PriceAreaChart data={chartData} anomalyPoints={anomalyPts} />
        }
      </Card>

      {/* Risk Score Timeline */}
      <Card>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Risk Timeline</h2>
          <p className="text-text-secondary text-sm">Ensemble anomaly score over time</p>
        </div>
        {loading
          ? <div className="h-[200px] bg-surface rounded-lg animate-pulse" />
          : <RiskScoreChart data={chartData} />
        }
      </Card>

      {/* Anomaly Events + Model Consensus */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-6">
        <Card>
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-text-primary">Anomaly Events</h2>
            <p className="text-text-secondary text-sm">{anomCount} anomalies detected in history</p>
          </div>
          <AnomalyTable
            events={historical?.events ?? []}
            loading={loading}
            maxRows={8}
          />
        </Card>

        {/* Forecast + Model Consensus */}
        <div className="space-y-6">
          {/* Tomorrow's Price */}
          <Card>
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wide">Tomorrow Prediction</h3>
            </div>
            {loading ? (
              <div className="h-16 bg-surface rounded animate-pulse" />
            ) : (
              <div className="flex items-baseline gap-3">
                <span className="font-mono font-bold text-3xl text-text-primary">
                  {formatPrice(tomorrowPrice)}
                </span>
                <span className={clsx(
                  'flex items-center gap-1 text-sm font-mono font-medium',
                  tomorrowChange >= 0 ? 'text-risk-normal' : 'text-risk-extreme'
                )}>
                  {tomorrowChange >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                  {formatPct(tomorrowChange)}
                </span>
              </div>
            )}
          </Card>

          {/* Model Consensus */}
          <ModelConsensus current={current} />
        </div>
      </div>
    </div>
  )
}
