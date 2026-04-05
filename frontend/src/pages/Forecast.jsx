import { useEffect, useMemo, useState } from 'react'
import { AlertTriangle, Loader2, TrendingDown, TrendingUp } from 'lucide-react'
import {
  Area,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts'
import clsx from 'clsx'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import FeatureImportanceChart from '../components/charts/FeatureImportanceChart'
import { formatDate, formatPct, formatPrice } from '../utils/formatters'
import {
  fetchHistoricalAnomalies,
  fetchLstmForecast,
  fetchPriceForecast,
  fetchTransformerForecast,
  fetchXgboostForecast,
} from '../services/api'

const FORECAST_METHODS = [
  { id: 'auto', label: 'Auto (Best)', category: 'Classical' },
  { id: 'naive', label: 'Naive', category: 'Classical' },
  { id: 'arima', label: 'ARIMA', category: 'Classical' },
  { id: 'ses', label: 'Exp. Smoothing', category: 'Classical' },
  { id: 'holt', label: "Holt's Linear", category: 'Classical' },
  { id: 'lstm', label: 'LSTM Seq2Seq', category: 'Deep Learning' },
  { id: 'transformer', label: 'Transformer', category: 'Deep Learning' },
  { id: 'xgboost', label: 'XGBoost', category: 'Gradient Boosting' },
]

function ForecastTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null

  const row = payload[0]?.payload
  return (
    <div className="min-w-[210px] rounded-xl border border-card-border bg-white p-4 shadow-float">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-text-muted">
        {label === 'Today' ? 'Today' : formatDate(label, 'MMM dd, yyyy')}
      </div>
      {row?.historical != null && (
        <div className="mb-1 flex items-center justify-between text-sm">
          <span className="text-text-secondary">Historical</span>
          <span className="font-mono font-semibold text-brand-blue">{formatPrice(row.historical)}</span>
        </div>
      )}
      {row?.forecast != null && (
        <div className="mb-1 flex items-center justify-between text-sm">
          <span className="text-text-secondary">Forecast</span>
          <span className="font-mono font-semibold text-chart-blue">{formatPrice(row.forecast)}</span>
        </div>
      )}
      {row?.lower != null && row?.upper != null && (
        <div className="mt-2 border-t border-card-border pt-2 text-xs text-text-secondary">
          95% CI: <span className="font-mono">{formatPrice(row.lower)} to {formatPrice(row.upper)}</span>
        </div>
      )}
    </div>
  )
}

export default function Forecast({ priceForecast: initialPriceForecast, selectedAsset, loading: parentLoading }) {
  const [selectedMethod, setSelectedMethod] = useState('auto')
  const [methodForecast, setMethodForecast] = useState(null)
  const [historyData, setHistoryData] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!selectedAsset) return

    fetchHistoricalAnomalies(selectedAsset, 120)
      .then((hist) => {
        setHistoryData((hist?.chart_data ?? []).slice(-90))
      })
      .catch(() => setHistoryData([]))
  }, [selectedAsset])

  useEffect(() => {
    if (!selectedAsset) return

    if (selectedMethod === 'auto') {
      return
    }

    let promise
    if (selectedMethod === 'lstm') {
      promise = fetchLstmForecast(selectedAsset, 30)
    } else if (selectedMethod === 'transformer') {
      promise = fetchTransformerForecast(selectedAsset, 30)
    } else if (selectedMethod === 'xgboost') {
      promise = fetchXgboostForecast(selectedAsset, 30)
    } else {
      promise = fetchPriceForecast(selectedAsset, 30, selectedMethod)
    }

    promise
      .then((data) => {
        setMethodForecast(data)
        setError(null)
      })
      .catch((err) => {
        setError(err?.message || 'Failed to load forecast')
      })
  }, [selectedMethod, selectedAsset, initialPriceForecast])

  const priceForecast = selectedMethod === 'auto' ? initialPriceForecast : methodForecast
  const loading = selectedMethod === 'auto' ? parentLoading : !methodForecast && !error

  const currentPrice = priceForecast?.current_price ?? 0
  const forecastValues = priceForecast?.forecast?.values ?? []
  const method = priceForecast?.method ?? 'auto'
  const horizon = priceForecast?.horizon ?? 30
  const summary = priceForecast?.summary ?? {}

  const methodInfo = FORECAST_METHODS.find((m) => m.id === selectedMethod)

  const chartData = useMemo(() => {
    const forecastSeries = priceForecast?.forecast ?? {}
    const values = forecastSeries?.values ?? []
    const dates = forecastSeries?.dates ?? []
    const lower = forecastSeries?.lower_95 ?? []
    const upper = forecastSeries?.upper_95 ?? []

    const historicalRows = historyData.map((point) => ({
      date: point.date,
      historical: point.close,
      forecast: null,
      lower: null,
      upper: null,
    }))

    const forecastRows = values.map((value, idx) => ({
      date: dates[idx] || `Day ${idx + 1}`,
      historical: null,
      forecast: value,
      lower: lower[idx] ?? value,
      upper: upper[idx] ?? value,
    }))

    if (historicalRows.length && forecastRows.length) {
      forecastRows[0].historical = historicalRows[historicalRows.length - 1].historical
    }

    return [...historicalRows, ...forecastRows]
  }, [historyData, priceForecast])

  const yValues = chartData.flatMap((row) => [row.historical, row.forecast, row.lower, row.upper]).filter(Number.isFinite)
  const yDomain = yValues.length
    ? [Math.min(...yValues) * 0.985, Math.max(...yValues) * 1.015]
    : ['auto', 'auto']

  const forecast30d = summary.forecast_30d ?? forecastValues[forecastValues.length - 1] ?? currentPrice
  const expectedReturn = summary.expected_return_pct ?? 0
  const forecast1d = forecastValues[0] ?? currentPrice
  const forecast7d = forecastValues[6] ?? forecastValues[forecastValues.length - 1] ?? currentPrice
  const isPositive = expectedReturn >= 0

  if (loading && !priceForecast) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Deep-Dive Forecasting</h1>
          <p className="text-sm text-text-secondary">Transformer, LSTM, and classical model projection center.</p>
        </div>
        <div className="flex h-[360px] items-center justify-center">
          <Loader2 size={34} className="animate-spin text-brand-blue" />
        </div>
      </div>
    )
  }

  if (error && !priceForecast) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Deep-Dive Forecasting</h1>
          <p className="text-sm text-text-secondary">Transformer, LSTM, and classical model projection center.</p>
        </div>
        <Card className="border-red-200 bg-red-50">
          <div className="flex items-center gap-3 text-red-700">
            <AlertTriangle size={20} />
            <span>Failed to load forecast data. Please try again.</span>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Deep-Dive Forecasting</h1>
          <p className="text-sm text-text-secondary">
            {selectedAsset} projection engine using {methodInfo?.label || method.toUpperCase()} with confidence envelope.
          </p>
        </div>
        <Badge variant="blue" className="text-xs">
          {horizon}-day horizon
        </Badge>
      </div>

      <Card className="bg-gradient-to-br from-white to-sky-50/50">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-text-primary">Model Selection</h3>
        <div className="space-y-3">
          {['Classical', 'Deep Learning', 'Gradient Boosting'].map((category) => (
            <div key={category}>
              <p className="mb-2 text-xs uppercase tracking-[0.14em] text-text-muted">{category}</p>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                {FORECAST_METHODS.filter((item) => item.category === category).map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setSelectedMethod(item.id)}
                    className={clsx(
                      'rounded-lg px-3 py-2 text-xs font-semibold transition-all',
                      selectedMethod === item.id
                        ? 'bg-brand-blue text-white shadow-glass'
                        : 'border border-card-border bg-white text-text-secondary hover:border-brand-blue/40 hover:text-text-primary'
                    )}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Card>
          <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-text-muted">Current Price</div>
          <div className="font-mono text-2xl font-bold text-text-primary">{formatPrice(currentPrice)}</div>
        </Card>
        <Card>
          <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-text-muted">Tomorrow</div>
          <div className="font-mono text-2xl font-bold text-text-primary">{formatPrice(forecast1d)}</div>
        </Card>
        <Card>
          <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-text-muted">7-Day Target</div>
          <div className="font-mono text-2xl font-bold text-text-primary">{formatPrice(forecast7d)}</div>
        </Card>
        <Card className={isPositive ? 'border-emerald-200 bg-emerald-50' : 'border-red-200 bg-red-50'}>
          <div className="mb-2 text-[11px] uppercase tracking-[0.14em] text-text-muted">30-Day Outlook</div>
          <div className={clsx('flex items-center gap-2 font-mono text-2xl font-bold', isPositive ? 'text-emerald-700' : 'text-red-700')}>
            <span>{formatPct(expectedReturn)}</span>
            {isPositive ? <TrendingUp size={18} /> : <TrendingDown size={18} />}
          </div>
          <div className="mt-1 text-xs text-text-secondary">Target {formatPrice(forecast30d)}</div>
        </Card>
      </div>

      <Card>
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-text-primary">Master Forecast Chart</h2>
            <p className="text-sm text-text-secondary">Historical trend, predicted trajectory, and 95% confidence interval.</p>
          </div>
          <div className="text-xs text-text-secondary">
            <span className="mr-4 inline-flex items-center gap-1"><span className="h-0.5 w-4 bg-brand-blue" />History</span>
            <span className="mr-4 inline-flex items-center gap-1"><span className="h-0.5 w-4 border-t-2 border-dashed border-chart-blue" />Forecast</span>
            <span className="inline-flex items-center gap-1"><span className="h-2 w-4 rounded bg-chart-blue/20" />95% CI</span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
            <defs>
              <linearGradient id="forecastBand" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#1D6FDC" stopOpacity={0.24} />
                <stop offset="100%" stopColor="#1D6FDC" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#DBE4EF" strokeDasharray="4 4" vertical={false} />
            <XAxis
              dataKey="date"
              tickFormatter={(value) => (value === 'Today' ? value : formatDate(value))}
              tick={{ fill: '#7C8BA1', fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={yDomain}
              tick={{ fill: '#7C8BA1', fontSize: 11 }}
              tickFormatter={(value) => formatPrice(value)}
              tickLine={false}
              axisLine={false}
              width={80}
            />
            <Tooltip content={<ForecastTooltip />} />

            <Area type="monotone" dataKey="upper" stroke="none" fill="url(#forecastBand)" connectNulls />
            <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" connectNulls />

            <Line
              type="monotone"
              dataKey="historical"
              stroke="#0B3A63"
              strokeWidth={2.6}
              dot={false}
              connectNulls
              name="Historical"
            />
            <Line
              type="monotone"
              dataKey="forecast"
              stroke="#1D6FDC"
              strokeWidth={2.4}
              strokeDasharray="6 4"
              dot={false}
              connectNulls
              name="Forecast"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </Card>

      {(selectedMethod === 'xgboost' || selectedMethod === 'transformer') && priceForecast?.feature_importance && (
        <FeatureImportanceChart
          importance={priceForecast.feature_importance}
          title={selectedMethod === 'transformer' ? 'Variable Selection Viewer' : 'Feature Influence Viewer'}
        />
      )}
    </div>
  )
}
