import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import clsx from 'clsx'
import {
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import { Card } from '../components/ui/Card'
import { API_BASE } from '../constants/config'
import { formatDate, formatPrice } from '../utils/formatters'

const MODEL_CONFIG = {
  baseline: [
    { id: 'zscore', name: 'Z-Score', desc: 'Statistical deviation detector' },
    { id: 'iforest', name: 'Isolation Forest', desc: 'Unsupervised anomaly detector' },
    { id: 'lstm', name: 'LSTM', desc: 'Deep sequence model' },
    { id: 'prophet', name: 'Prophet', desc: 'Trend and seasonality residuals' },
  ],
  advanced: [
    { id: 'xgb', name: 'XGBoost', desc: 'Supervised crash predictor' },
    { id: 'hmm', name: 'HMM Regime', desc: 'Market-state classifier' },
    { id: 'tcn', name: 'TCN', desc: 'Temporal convolutional sequence model' },
  ],
}

function getRiskStyles(score) {
  if (score < 40) return 'text-emerald-700 bg-emerald-50 border-emerald-200'
  if (score < 60) return 'text-amber-700 bg-amber-50 border-amber-200'
  if (score < 75) return 'text-orange-700 bg-orange-50 border-orange-200'
  return 'text-red-700 bg-red-50 border-red-200'
}

function OverlayTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  return (
    <div className="min-w-[190px] rounded-xl border border-card-border bg-white p-3 shadow-float">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-text-muted">
        {formatDate(label, 'MMM dd, yyyy')}
      </div>
      <div className="flex items-center justify-between text-sm">
        <span className="text-text-secondary">Price</span>
        <span className="font-mono font-semibold text-brand-blue">{formatPrice(row?.close)}</span>
      </div>
      {row?.isAnomaly && (
        <div className="mt-2 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-red-700">
          Anomaly Zone
        </div>
      )}
    </div>
  )
}

function AgreementGauge({ agreement, total }) {
  const progress = total > 0 ? agreement / total : 0
  const radius = 45
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - progress)

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width="120" height="120" viewBox="0 0 120 120">
        <circle cx="60" cy="60" r={radius} stroke="#DBE4EF" strokeWidth="10" fill="none" />
        <circle
          cx="60"
          cy="60"
          r={radius}
          stroke="#1D6FDC"
          strokeWidth="10"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 60 60)"
        />
        <text x="60" y="56" textAnchor="middle" className="fill-text-primary font-mono text-lg font-bold">
          {agreement}/{total}
        </text>
        <text x="60" y="72" textAnchor="middle" className="fill-text-muted text-[10px] uppercase tracking-[0.1em]">
          Models
        </text>
      </svg>
      <p className="text-center text-xs text-text-secondary">Models currently signaling elevated anomaly risk.</p>
    </div>
  )
}

export default function AdvancedAnomaly({ selectedAsset, loading: parentLoading }) {
  const [advancedData, setAdvancedData] = useState(null)
  const [baselineData, setBaselineData] = useState(null)
  const [historyData, setHistoryData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!selectedAsset) return

    Promise.all([
      fetch(`${API_BASE}/anomaly/advanced/${selectedAsset}`).then((r) => r.json()),
      fetch(`${API_BASE}/anomaly/current/${selectedAsset}`).then((r) => r.json()),
      fetch(`${API_BASE}/anomaly/historical/${selectedAsset}?top_n=180`).then((r) => r.json()),
    ])
      .then(([advanced, baseline, history]) => {
        setAdvancedData(advanced)
        setBaselineData(baseline)
        setHistoryData(history)
        setError(null)
      })
      .catch((err) => setError(err?.message || 'Failed to load data'))
  }, [selectedAsset])

  const isLoading = (!advancedData && !baselineData && !historyData) || parentLoading
  const advScore = advancedData?.advanced_ensemble ?? 0
  const baseScore = baselineData?.ensemble_score ?? 0
  const regimeText = advancedData?.current_regime ?? 'unknown'

  const modelScores = advancedData?.model_scores ?? {}
  const scoreValues = Object.values(modelScores).filter((v) => Number.isFinite(v))
  const agreementCount = scoreValues.filter((value) => value >= 60).length

  const chartData = useMemo(() => {
    const anomalyDates = new Set((historyData?.events ?? []).map((event) => event.date))
    return (historyData?.chart_data ?? []).slice(-160).map((row) => ({
      ...row,
      isAnomaly: anomalyDates.has(row.date),
    }))
  }, [historyData])

  const priceValues = chartData.map((row) => row.close).filter(Number.isFinite)
  const yDomain = priceValues.length
    ? [Math.min(...priceValues) * 0.985, Math.max(...priceValues) * 1.015]
    : ['auto', 'auto']

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Advanced Anomaly Detection</h1>
          <p className="text-sm text-text-secondary">7-model ensemble with structural break monitoring.</p>
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
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Advanced Anomaly Detection</h1>
          <p className="text-sm text-text-secondary">Ensemble structural stress detection for {selectedAsset}.</p>
        </div>
        <div className={clsx('rounded-lg border px-4 py-2 text-sm font-semibold uppercase', getRiskStyles(advScore))}>
          {regimeText} regime
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card>
          <p className="text-xs uppercase tracking-[0.14em] text-text-muted">Baseline Ensemble</p>
          <p className="font-mono text-3xl font-bold text-text-primary">{baseScore.toFixed(1)}</p>
          <p className="mt-2 text-xs text-text-secondary">4-model signal</p>
        </Card>
        <Card className="border-brand-blue/20 bg-gradient-to-br from-white to-sky-50">
          <p className="text-xs uppercase tracking-[0.14em] text-text-muted">Advanced Ensemble</p>
          <p className="font-mono text-3xl font-bold text-brand-blue">{advScore.toFixed(1)}</p>
          <p className="mt-2 text-xs text-text-secondary">7-model signal</p>
        </Card>
        <Card>
          <p className="text-xs uppercase tracking-[0.14em] text-text-muted">Delta</p>
          <p className={clsx('font-mono text-3xl font-bold', advScore - baseScore >= 0 ? 'text-red-600' : 'text-emerald-600')}>
            {advScore - baseScore >= 0 ? '+' : ''}{(advScore - baseScore).toFixed(1)}
          </p>
          <p className="mt-2 text-xs text-text-secondary">advanced minus baseline</p>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.3fr_0.7fr]">
        <Card>
          <h2 className="mb-1 text-lg font-semibold text-text-primary">Anomaly Overlay Chart</h2>
          <p className="mb-4 text-sm text-text-secondary">Danger blocks mark days tagged by the ensemble as anomalous.</p>
          {isLoading ? (
            <div className="h-[320px] animate-pulse rounded-lg bg-surface" />
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
                <CartesianGrid stroke="#DBE4EF" strokeDasharray="4 4" vertical={false} />
                <XAxis dataKey="date" tickFormatter={(value) => formatDate(value)} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} />
                <YAxis domain={yDomain} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(v) => formatPrice(v)} width={75} />
                <Tooltip content={<OverlayTooltip />} />
                {chartData.filter((row) => row.isAnomaly).map((row) => (
                  <ReferenceArea key={`anom-${row.date}`} x1={row.date} x2={row.date} fill="#ef4444" fillOpacity={0.14} />
                ))}
                <Line type="monotone" dataKey="close" stroke="#0B3A63" strokeWidth={2.4} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card>
          <h2 className="mb-2 text-lg font-semibold text-text-primary">Model Agreement Gauge</h2>
          <p className="mb-3 text-sm text-text-secondary">Consensus strength across all 7 models.</p>
          <AgreementGauge agreement={agreementCount} total={7} />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[...MODEL_CONFIG.baseline, ...MODEL_CONFIG.advanced].map((model) => {
          const value = modelScores?.[model.id] ?? 0
          return (
            <Card key={model.id} className="border-card-border/90">
              <p className="text-sm font-semibold text-text-primary">{model.name}</p>
              <p className="mt-1 text-xs text-text-secondary">{model.desc}</p>
              <p className="mt-3 font-mono text-2xl font-bold text-brand-blue">{value.toFixed(1)}</p>
            </Card>
          )
        })}
      </div>

      <Card>
        <h3 className="mb-3 text-lg font-semibold text-text-primary">Interpretation</h3>
        <div className="grid grid-cols-1 gap-4 text-sm text-text-secondary md:grid-cols-2">
          <div>
            <p className="font-semibold text-text-primary">Signal Bias</p>
            <p className="mt-1">High agreement and positive ensemble deltas suggest broad multi-model stress confirmation.</p>
          </div>
          <div>
            <p className="font-semibold text-text-primary">Regime Context</p>
            <p className="mt-1">Regime state calibrates severity. Crisis regime plus high agreement requires immediate risk review.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
