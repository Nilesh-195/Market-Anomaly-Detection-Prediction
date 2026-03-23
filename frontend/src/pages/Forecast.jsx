import { useState, useEffect } from 'react'
import {
  TrendingUp, TrendingDown, Activity, Calendar, BarChart3,
  ChevronRight, Loader2, AlertTriangle,
} from 'lucide-react'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import ForecastChart from '../components/charts/ForecastChart'
import MethodComparisonTable from '../components/widgets/MethodComparisonTable'
import { formatPrice, formatPct, formatDate } from '../utils/formatters'
import clsx from 'clsx'

export default function Forecast({ priceForecast, selectedAsset, loading, error }) {
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

  if (loading) {
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

  if (error) {
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Price Forecast</h1>
          <p className="text-text-secondary text-sm">
            {selectedAsset} predictions using {method.toUpperCase()} method
          </p>
        </div>
        <Badge variant="blue" className="text-xs">
          {horizon}-day horizon
        </Badge>
      </div>

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
              <Badge variant="blue">{method.toUpperCase()}</Badge>
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
    </div>
  )
}
