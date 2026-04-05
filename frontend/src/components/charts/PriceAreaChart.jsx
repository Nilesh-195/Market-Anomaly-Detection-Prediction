import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceArea,
} from 'recharts'
import { formatDate, formatPrice, formatScore } from '../../utils/formatters'
import { getRiskColor } from '../../utils/riskHelpers'
import { COLOURS } from '../../constants/colours'

function formatAxisPrice(value) {
  if (value == null || Number.isNaN(value)) return '$0'
  const abs = Math.abs(value)
  if (abs >= 1000) return `$${(value / 1000).toFixed(1)}k`
  if (abs >= 100) return `$${value.toFixed(0)}`
  return `$${value.toFixed(2)}`
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="bg-white border border-card-border rounded-xl p-4 shadow-float animate-float-up min-w-[200px]">
      <div className="text-text-secondary text-xs uppercase tracking-wider mb-3 font-semibold">{formatDate(label, 'MMM dd, yyyy')}</div>
      <div className="space-y-2">
        {payload.map((p, i) => (
          <div key={i} className="flex justify-between items-center gap-6 text-sm border-b border-card-border pb-2">
            <span className="text-text-secondary">{p.name}</span>
            <span className="font-mono font-bold text-base" style={{ color: p.color }}>
              {p.name === 'Price' ? formatPrice(p.value) : formatScore(p.value)}
            </span>
          </div>
        ))}
        {d?.risk_score != null && (
          <div className="flex justify-between items-center gap-6 text-xs text-text-muted pt-1">
            <span className="uppercase tracking-wider">Risk Score</span>
            <span className="font-mono font-medium" style={{ color: getRiskColor(d.risk_score) }}>
              {formatScore(d.risk_score)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export default function PriceAreaChart({ data = [], anomalyPoints = [] }) {
  // Build a set of anomaly dates for fast lookup
  const anomalyDates = new Set(anomalyPoints.map(a => a.date))
  // For ReferenceDots we need the actual close value from chart_data
  const anomalyDots = data.filter(d => anomalyDates.has(d.date) && d.close != null)
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={COLOURS.chartBlue} stopOpacity={0.25} />
            <stop offset="95%" stopColor={COLOURS.chartBlue} stopOpacity={0}    />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={COLOURS.cardBorder} vertical={false} />
        <XAxis
          dataKey="date"
          tickFormatter={v => formatDate(v)}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          orientation="right"
          tickFormatter={formatAxisPrice}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} width={64}
          domain={['auto', 'auto']}
        />
        <Tooltip content={<CustomTooltip />} />
        
        {/* Draw Anomaly Bounds FIRST so area lines go over them */}
        {anomalyDots.map((pt, i) => (
          <ReferenceArea
            key={`anomaly-${i}`}
            x1={pt.date}
            x2={pt.date}
            fill={COLOURS.riskExtreme}
            fillOpacity={0.15}
          />
        ))}

        <Area
          type="monotone" dataKey="close" name="Price"
          stroke={COLOURS.chartBlue} strokeWidth={2.5}
          fill="url(#priceGrad)" dot={false} 
          activeDot={{ r: 6, fill: COLOURS.chartBlue, stroke: '#fff', strokeWidth: 2, className: 'shadow-glass' }}
          animationDuration={1200}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
