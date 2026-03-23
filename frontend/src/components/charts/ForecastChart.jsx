import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { formatDate, formatPrice } from '../../utils/formatters'
import { COLOURS } from '../../constants/colours'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-3 shadow-2xl min-w-[180px]">
      <div className="text-text-secondary text-xs mb-2">{formatDate(label, 'MMM dd, yyyy')}</div>
      <div className="space-y-1">
        <div className="flex justify-between gap-4 text-sm">
          <span className="text-text-secondary">Forecast</span>
          <span className="font-mono font-semibold text-brand-blue">
            {formatPrice(d?.forecast)}
          </span>
        </div>
        <div className="flex justify-between gap-4 text-xs text-text-muted">
          <span>95% CI</span>
          <span className="font-mono">
            {formatPrice(d?.lower)} — {formatPrice(d?.upper)}
          </span>
        </div>
      </div>
    </div>
  )
}

export default function ForecastChart({ data = [], currentPrice = 0 }) {
  if (!data.length) {
    return (
      <div className="h-[280px] flex items-center justify-center text-text-muted">
        No forecast data available
      </div>
    )
  }

  // Add current price as first point
  const chartData = [
    { date: 'Today', forecast: currentPrice, lower: currentPrice, upper: currentPrice },
    ...data,
  ]

  return (
    <ResponsiveContainer width="100%" height={280}>
      <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLOURS.brandBlue} stopOpacity={0.25} />
            <stop offset="95%" stopColor={COLOURS.brandBlue} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={COLOURS.brandBlue} stopOpacity={0.1} />
            <stop offset="95%" stopColor={COLOURS.brandBlue} stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={COLOURS.cardBorder} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={v => v === 'Today' ? v : formatDate(v)}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          orientation="right"
          tickFormatter={v => `$${(v / 1000).toFixed(1)}k`}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false}
          tickLine={false}
          width={55}
          domain={['dataMin - 100', 'dataMax + 100']}
        />
        <Tooltip content={<CustomTooltip />} />

        {/* Current price reference line */}
        <ReferenceLine
          y={currentPrice}
          stroke={COLOURS.textMuted}
          strokeDasharray="4 4"
          strokeOpacity={0.5}
          label={{
            value: 'Current',
            fill: COLOURS.textMuted,
            fontSize: 10,
            position: 'left',
          }}
        />

        {/* Confidence interval band */}
        <Area
          type="monotone"
          dataKey="upper"
          stroke="none"
          fill="url(#ciGrad)"
          animationDuration={800}
        />
        <Area
          type="monotone"
          dataKey="lower"
          stroke="none"
          fill={COLOURS.pageBg}
          animationDuration={800}
        />

        {/* Main forecast line */}
        <Area
          type="monotone"
          dataKey="forecast"
          name="Forecast"
          stroke={COLOURS.brandBlue}
          strokeWidth={2}
          fill="url(#forecastGrad)"
          dot={false}
          activeDot={{ r: 5, fill: COLOURS.brandBlue, stroke: '#fff', strokeWidth: 2 }}
          animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
