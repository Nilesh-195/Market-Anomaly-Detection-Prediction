import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceDot,
} from 'recharts'
import { formatDate, formatPrice, formatScore } from '../../utils/formatters'
import { getRiskColor } from '../../utils/riskHelpers'
import { COLOURS } from '../../constants/colours'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-3 shadow-2xl min-w-[180px]">
      <div className="text-[#64748B] text-xs mb-2">{formatDate(label, 'MMM dd, yyyy')}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex justify-between gap-4 text-xs">
          <span className="text-[#64748B]">{p.name}</span>
          <span className="font-mono font-medium" style={{ color: p.color }}>
            {p.name === 'Price' ? formatPrice(p.value) : formatScore(p.value)}
          </span>
        </div>
      ))}
      {d?.risk_score != null && (
        <div className="flex justify-between gap-4 text-xs mt-1 pt-1 border-t border-card-border">
          <span className="text-[#64748B]">Risk</span>
          <span className="font-mono font-medium" style={{ color: getRiskColor(d.risk_score) }}>
            {formatScore(d.risk_score)}
          </span>
        </div>
      )}
    </div>
  )
}

export default function PriceAreaChart({ data = [], anomalyPoints = [] }) {
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
          tickFormatter={v => `$${(v/1000).toFixed(0)}k`}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} width={55}
        />
        <Tooltip content={<CustomTooltip />} />
        <Area
          type="monotone" dataKey="close" name="Price"
          stroke={COLOURS.chartBlue} strokeWidth={1.5}
          fill="url(#priceGrad)" dot={false} activeDot={{ r: 4 }}
          animationDuration={800}
        />
        {anomalyPoints.map((pt, i) => (
          <ReferenceDot
            key={i} x={pt.date} y={pt.close}
            r={4} fill={COLOURS.riskExtreme}
            stroke={COLOURS.cardBg} strokeWidth={2}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  )
}
