import { useState, useEffect } from 'react'
import {
  TrendingUp, TrendingDown, Activity, Calendar, BarChart3,
  ChevronRight, Loader2, AlertTriangle,
} from 'lucide-react'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import ForecastChart from '../components/charts/ForecastChart'
import AttentionHeatmap from '../components/charts/AttentionHeatmap'
import FeatureImportanceChart from '../components/charts/FeatureImportanceChart'
import { formatPrice, formatPct, formatDate } from '../utils/formatters'
import { fetchLstmForecast, fetchTransformerForecast, fetchXgboostForecast, fetchPriceForecast } from '../services/api'
import clsx from 'clsx'

const FORECAST_METHODS = [
  { id: 'auto', label: 'Auto (Best)', category: 'Classical' },
  { id: 'naive', label: 'Naive', category: 'Classical' },
  { id: 'arima', label: 'ARIMA', category: 'Classical' },
  { id: 'ses', label: 'Exp. Smoothing', category: 'Classical' },
  { id: 'holt', label: 'Holt\'s Linear', category: 'Classical' },
  { id: 'lstm', label: 'LSTM Seq2Seq', category: 'Deep Learning' },
  { id: 'transformer', label: 'Transformer', category: 'Deep Learning' },
  { id: 'xgboost', label: 'XGBoost', category: 'Gradient Boosting' },
]

export default function Forecast({ priceForecast: initialPriceForecast, selectedAsset, loading: parentLoading }) {
  const [selectedMethod, setSelectedMethod] = useState('auto')
  const [priceForecast, setPriceForecast] = useState(initialPriceForecast)
  const [loading, setLoading] = useState(parentLoading)
  const [error, setError] = useState(null)

  // Load forecast data when method or asset changes
  useEffect(() => {
    if (!selectedAsset) return

    if (selectedMethod === 'auto') {
      setPriceForecast(initialPriceForecast)
      setLoading(parentLoading)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)

    let promise
    if (selectedMethod === 'lstm') {
      promise = fetchLstmForecast(selectedAsset, 30)
    } else if (selectedMethod === 'transformer') {
      promise = fetchTransformerForecast(selectedAsset, 30)
    } else if (selectedMethod === 'xgboost') {
      promise = fetchXgboostForecast(selectedAsset, 30)
    } else {
      // Classical methods: naive, arima, ses, holt — use the price endpoint with method param
      promise = fetchPriceForecast(selectedAsset, 30, selectedMethod)
    }

    promise
      .then(data => {
        setPriceForecast(data)
        setError(null)
      })
      .catch(err => {
        setError(err?.message || 'Failed to load forecast')
        if (initialPriceForecast) setPriceForecast(initialPriceForecast)
      })
      .finally(() => setLoading(false))
  }, [selectedMethod, selectedAsset])

  // Sync when parent's initial forecast data arrives
  useEffect(() => {
    if (selectedMethod === 'auto' && initialPriceForecast) {
      setPriceForecast(initialPriceForecast)
      setLoading(false)
      setError(null)
    }
  }, [initialPriceForecast])

  // Reflect parent loading state only for 'auto' method
  useEffect(() => {
    if (selectedMethod === 'auto') {
      setLoading(parentLoading)
    }
  }, [parentLoading, selectedMethod])

  // Self-fetch fallback: if parent gave no data and finished loading, fetch directly
  useEffect(() => {
    if (!parentLoading && !initialPriceForecast && selectedMethod === 'auto' && selectedAsset) {
      setLoading(true)
      setError(null)
      fetchPriceForecast(selectedAsset, 30, 'auto')
        .then(data => {
          setPriceForecast(data)
          setError(null)
        })
        .catch(err => {
          setError(err?.message || 'Failed to load forecast data')
        })
        .finally(() => setLoading(false))
    }
  }, [parentLoading, initialPriceForecast, selectedAsset, selectedMethod])

  const currentPrice = priceForecast?.current_price ?? 0
  const forecastValues = priceForecast?.forecast?.values ?? []
  const forecastDates = priceForecast?.forecast?.dates ?? []
  const lowerBand = priceForecast?.forecast?.lower_95 ?? []
  const upperBand = priceForecast?.forecast?.upper_95 ?? []
  const method = priceForecast?.method ?? 'auto'
  const horizon = priceForecast?.horizon ?? 30
  const summary = priceForecast?.summary ?? {}

  // Build chart data
  const chartData = forecastValues.map((val, i) => ({
    date: forecastDates[i] || `Day ${i + 1}`,
    forecast: val,
    lower: lowerBand[i] ?? val * 0.95,
    upper: upperBand[i] ?? val * 1.05,
  }))

  // Key metrics
  const forecast30d = summary.forecast_30d ?? forecastValues[forecastValues.length - 1] ?? currentPrice
  const expectedReturn = summary.expected_return_pct ?? 0
  const forecast1d = forecastValues[0] ?? currentPrice
  const forecast7d = forecastValues[6] ?? forecastValues[forecastValues.length - 1] ?? currentPrice

  const isPositive = expectedReturn >= 0

  if (loading && !priceForecast) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Price Forecast</h1>
          <p className="text-text-secondary text-sm">AI-powered price predictions with confidence intervals</p>
        </div>
        <div className="flex items-center justify-center h-[400px]">
          <Loader2 size={32} className="animate-spin text-brand-blue" />
        </div>
      </div>
    )
  }

  if (error && !priceForecast) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Price Forecast</h1>
          <p className="text-text-secondary text-sm">AI-powered price predictions with confidence intervals</p>
        </div>
        <Card className="bg-red-50 border-red-200">
          <div className="flex items-center gap-3 text-red-600">
            <AlertTriangle size={20} />
            <span>Failed to load forecast data. Please try again.</span>
          </div>
        </Card>
      </div>
    )
  }

  const methodInfo = FORECAST_METHODS.find(m => m.id === selectedMethod)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Price Forecast</h1>
          <p className="text-text-secondary text-sm">
            {selectedAsset} predictions using {methodInfo?.label || method.toUpperCase()} method
          </p>
        </div>
        <Badge variant="blue" className="text-xs">
          {horizon}-day horizon
        </Badge>
      </div>

      {/* Method Selector */}
      <Card>
        <div className="mb-4">
          <h3 className="text-sm font-medium text-text-primary uppercase tracking-wide mb-3">Forecast Method</h3>
          <div className="space-y-3">
            {['Classical', 'Deep Learning', 'Gradient Boosting'].map(category => (
              <div key={category}>
                <p className="text-xs text-text-secondary uppercase tracking-wider mb-2">{category}</p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {FORECAST_METHODS.filter(m => m.category === category).map(method => (
                    <button
                      key={method.id}
                      onClick={() => setSelectedMethod(method.id)}
                      className={clsx(
                        'px-3 py-2 rounded-lg text-xs font-medium transition-colors',
                        selectedMethod === method.id
                          ? 'bg-brand-blue text-white'
                          : 'bg-surface hover:bg-surface-alt text-text-secondary hover:text-text-primary'
                      )}
                    >
                      {method.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </Card>

      {/* Key Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <div className="text-text-secondary text-[11px] uppercase tracking-wider mb-2">Current Price</div>
          <div className="font-mono font-bold text-2xl text-text-primary">
            {formatPrice(currentPrice)}
          </div>
        </Card>

        <Card>
          <div className="text-text-secondary text-[11px] uppercase tracking-wider mb-2">Tomorrow</div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono font-bold text-2xl text-text-primary">
              {formatPrice(forecast1d)}
            </span>
            <span className={clsx(
              'text-sm font-mono',
              forecast1d >= currentPrice ? 'text-risk-normal' : 'text-risk-extreme'
            )}>
              {formatPct(((forecast1d / currentPrice) - 1) * 100)}
            </span>
          </div>
        </Card>

        <Card>
          <div className="text-text-secondary text-[11px] uppercase tracking-wider mb-2">7-Day</div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono font-bold text-2xl text-text-primary">
              {formatPrice(forecast7d)}
            </span>
            <span className={clsx(
              'text-sm font-mono',
              forecast7d >= currentPrice ? 'text-risk-normal' : 'text-risk-extreme'
            )}>
              {formatPct(((forecast7d / currentPrice) - 1) * 100)}
            </span>
          </div>
        </Card>

        <Card className={isPositive ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}>
          <div className="text-text-secondary text-[11px] uppercase tracking-wider mb-2">30-Day Outlook</div>
          <div className="flex items-baseline gap-2">
            <span className={clsx(
              'font-mono font-bold text-2xl',
              isPositive ? 'text-risk-normal' : 'text-risk-extreme'
            )}>
              {formatPct(expectedReturn)}
            </span>
            {isPositive ? <TrendingUp size={20} className="text-risk-normal" /> : <TrendingDown size={20} className="text-risk-extreme" />}
          </div>
          <div className="text-xs text-text-secondary mt-1">
            Target: {formatPrice(forecast30d)}
          </div>
        </Card>
      </div>

      {/* Forecast Chart */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-text-primary">Price Forecast</h2>
            <p className="text-text-secondary text-sm">Predicted values with 95% confidence interval</p>
          </div>
          <div className="flex items-center gap-4 text-xs text-text-secondary">
            <span className="flex items-center gap-1">
              <span className="w-3 h-0.5 bg-brand-blue rounded"></span>
              Forecast
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 bg-brand-blue/20 rounded"></span>
              95% CI
            </span>
          </div>
        </div>
        <ForecastChart data={chartData} currentPrice={currentPrice} />
      </Card>

      {/* Method Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-lg font-semibold text-text-primary mb-4">Forecast Method</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-card-border">
              <span className="text-text-secondary text-sm">Model</span>
              <Badge variant="blue">{methodInfo?.label || method.toUpperCase()}</Badge>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-card-border">
              <span className="text-text-secondary text-sm">Category</span>
              <Badge variant="blue" className="text-xs bg-blue-600/20 text-blue-300 border-blue-400">{methodInfo?.category || 'Unknown'}</Badge>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-card-border">
              <span className="text-text-secondary text-sm">Horizon</span>
              <span className="font-mono text-text-primary">{horizon} days</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-card-border">
              <span className="text-text-secondary text-sm">Confidence Level</span>
              <span className="font-mono text-text-primary">95%</span>
            </div>
            {priceForecast?.model_info?.aic && (
              <div className="flex items-center justify-between py-2">
                <span className="text-text-secondary text-sm">AIC Score</span>
                <span className="font-mono text-text-primary">
                  {priceForecast.model_info.aic.toFixed(2)}
                </span>
              </div>
            )}
            {priceForecast?.model_info?.type && (
              <div className="flex items-center justify-between py-2 pt-2">
                <span className="text-text-secondary text-sm">Model Type</span>
                <span className="font-mono text-xs text-text-secondary">{priceForecast.model_info.type}</span>
              </div>
            )}
          </div>
        </Card>

        <Card>
          <h3 className="text-lg font-semibold text-text-primary mb-4">Prediction Summary</h3>
          <div className="space-y-4">
            <div className="p-3 bg-surface rounded-lg">
              <div className="text-text-secondary text-xs mb-1">Short-term (1-7 days)</div>
              <div className={clsx(
                'font-medium',
                forecast7d >= currentPrice ? 'text-risk-normal' : 'text-risk-extreme'
              )}>
                {forecast7d >= currentPrice ? 'Bullish' : 'Bearish'} — Expecting {formatPct(((forecast7d / currentPrice) - 1) * 100)} move
              </div>
            </div>
            <div className="p-3 bg-surface rounded-lg">
              <div className="text-text-secondary text-xs mb-1">Medium-term (30 days)</div>
              <div className={clsx(
                'font-medium',
                isPositive ? 'text-risk-normal' : 'text-risk-extreme'
              )}>
                {isPositive ? 'Bullish' : 'Bearish'} — Expecting {formatPct(expectedReturn)} return
              </div>
            </div>
            <div className="text-xs text-text-muted mt-2">
              * Forecasts are probabilistic estimates. Actual returns may vary.
            </div>
          </div>
        </Card>
      </div>

      {/* LSTM Attention Weights */}
      {selectedMethod === 'lstm' && priceForecast?.attention_weights && (
        <AttentionHeatmap weights={priceForecast.attention_weights} title="LSTM Attention Weights" />
      )}

      {/* XGBoost Feature Importance */}
      {selectedMethod === 'xgboost' && priceForecast?.feature_importance && (
        <FeatureImportanceChart importance={priceForecast.feature_importance} title="XGBoost Feature Importance" />
      )}
    </div>
  )
}
