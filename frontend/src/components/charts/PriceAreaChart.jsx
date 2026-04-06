import { useMemo } from 'react'
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
    <div className="bg-card-bg border border-card-border rounded-xl p-4 shadow-lg min-w-[200px]">
      <div className="text-text-secondary text-xs uppercase tracking-wider mb-3 font-semibold">{formatDate(label, 'MMM dd, yyyy')}</div>
      <div className="space-y-2">
        {payload.map((p, i) => (
          <div key={i} className="flex justify-between items-center gap-6 text-sm border-b border-card-border pb-2">
            <span className="text-text-secondary">{p.name}</span>
            <span className="font-mono font-bold text-base" style={{ color: p.color || COLOURS.textPrimary }}>
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

export default function PriceAreaChart({ data = [], anomalyPoints = [], showWindows = true }) {
  
  // Compute window segments for ReferenceArea
  const anomalyWindows = useMemo(() => {
    if (!showWindows || !anomalyPoints.length || !data.length) return []
    
    return anomalyPoints.map(pt => {
      if (pt.start_date && pt.end_date) {
        return { x1: pt.start_date, x2: pt.end_date }
      }
      // Compute +/- 2 trading days if only 'date' is present
      const eventDate = pt.date
      const idx = data.findIndex(d => d.date === eventDate)
      
      if (idx === -1) return { x1: eventDate, x2: eventDate }

      const startIdx = Math.max(0, idx - 2)
      const endIdx = Math.min(data.length - 1, idx + 2)
      return { x1: data[startIdx].date, x2: data[endIdx].date }
    })
  }, [data, anomalyPoints, showWindows])

  return (
    <ResponsiveContainer width="100%" height={280}>
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
        <Tooltip content={<CustomTooltip />} cursor={{ stroke: COLOURS.textMuted, strokeWidth: 1, strokeDasharray: '4 4' }} />
        
        {/* Draw Anomaly Bounds FIRST so area lines go over them */}
        {anomalyWindows.map((win, i) => (
          <ReferenceArea
            key={`anomaly-win-${i}`}
            x1={win.x1}
            x2={win.x2}
            fill="#EF4444" // Soft red/amber tint
            fillOpacity={0.15}
            strokeOpacity={0}
          />
        ))}

        <Area
          type="monotone" dataKey="close" name="Price"
          stroke={COLOURS.chartBlue} strokeWidth={2.5}
          fill="url(#priceGrad)" dot={false} 
          activeDot={{ r: 6, fill: COLOURS.chartBlue, stroke: '#fff', strokeWidth: 2, className: 'shadow-lg' }}
          animationDuration={1000}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
