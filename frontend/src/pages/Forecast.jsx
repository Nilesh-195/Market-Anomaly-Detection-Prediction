import { motion } from 'framer-motion'
import { TrendingUp, AlertTriangle } from 'lucide-react'
import { Card } from '../components/ui/Card'
import ForecastDayCard from '../components/cards/ForecastDayCard'
import ForecastChart from '../components/charts/ForecastChart'
import { Badge } from '../components/ui/Badge'
import { formatScore } from '../utils/formatters'
import { getRiskColor } from '../utils/riskHelpers'

export default function Forecast({ forecast, historical, loading }) {
  const forecastDays = forecast?.forecast ?? []
  const histData     = (historical?.chart_data ?? []).slice(-30).map(d => ({
    ...d,
    ensemble_score: d.ensemble_score ?? null,
    forecast_score: null,
  }))
  const foreData = forecastDays.map((d, i) => ({
    date:           d.date,
    ensemble_score: null,
    forecast_score: d.mean,
    ci:             [d.lower ?? d.mean - 5, d.upper ?? d.mean + 5],
  }))
  const combined = [...histData, ...foreData]

  const maxForecast = forecastDays.reduce((m, d) => Math.max(m, d.mean ?? 0), 0)
  const currentScore = histData[histData.length - 1]?.ensemble_score ?? 0
  const volatilityWarning = maxForecast > currentScore * 1.5 && maxForecast > 50

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-[#F1F5F9] text-xl font-semibold">5-Day Anomaly Forecast</h1>
          <p className="text-[#64748B] text-sm mt-0.5">ARIMA model · LSTM pattern recognition</p>
        </div>
        <Badge variant="yellow">Not financial advice</Badge>
      </div>

      {/* Volatility warning */}
      {volatilityWarning && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="flex items-center gap-4 bg-[#1a1200] border-l-4 border-risk-elevated rounded-xl p-4"
        >
          <TrendingUp size={18} className="text-risk-elevated flex-shrink-0" />
          <div>
            <div className="text-[#F1F5F9] font-medium text-sm">Elevated anomaly score expected</div>
            <div className="text-[#64748B] text-xs">
              Current: {formatScore(currentScore)} → Forecast peak: {formatScore(maxForecast)}
            </div>
          </div>
        </motion.div>
      )}

      {/* Day cards */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
        {loading
          ? Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-36 bg-card-bg border border-card-border rounded-xl animate-pulse" />
            ))
          : forecastDays.slice(0, 5).map((d, i) => (
              <ForecastDayCard
                key={d.date}
                day={i === 0 ? 'Tomorrow' : `Day +${i + 1}`}
                date={d.date}
                score={d.mean}
                delta={d.mean - currentScore}
                index={i}
              />
            ))
        }
      </div>

      {/* Forecast chart */}
      <Card>
        <div className="text-[#F1F5F9] font-medium mb-1">Score Forecast</div>
        <div className="text-[#64748B] text-xs mb-4">
          Last 30 days (solid) + 5-day forecast (dashed) with confidence interval
        </div>
        {loading
          ? <div className="h-[260px] bg-surface rounded-lg animate-pulse" />
          : <ForecastChart historical={histData} forecast={foreData} />
        }
      </Card>

      {/* Risk breakdown table */}
      <Card>
        <div className="text-[#F1F5F9] font-medium mb-1">Forecast Risk Breakdown</div>
        <div className="text-[#64748B] text-xs mb-4">Per-day predicted anomaly score</div>
        <div className="space-y-3">
          {forecastDays.slice(0, 5).map((d, i) => {
            const score = d.mean ?? 0
            const color = getRiskColor(score)
            return (
              <div key={d.date} className="flex items-center gap-3">
                <span className="text-[#64748B] text-xs font-mono w-20">
                  {i === 0 ? 'Tomorrow' : `Day +${i + 1}`}
                </span>
                <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${score}%` }}
                    transition={{ delay: i * 0.1, duration: 0.6 }}
                    className="h-full rounded-full"
                    style={{ background: color }}
                  />
                </div>
                <span className="font-mono text-xs w-10 text-right" style={{ color }}>
                  {formatScore(score)}
                </span>
              </div>
            )
          })}
        </div>
      </Card>
    </div>
  )
}
