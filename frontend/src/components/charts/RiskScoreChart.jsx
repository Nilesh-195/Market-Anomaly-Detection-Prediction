import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, ReferenceDot,
} from 'recharts'
import { formatDate, formatScore } from '../../utils/formatters'
import { getRiskColor, getRiskLabel } from '../../utils/riskHelpers'
import { COLOURS } from '../../constants/colours'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const score = payload[0]?.value
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-3 shadow-2xl">
      <div className="text-[#64748B] text-xs mb-1">{formatDate(label, 'MMM dd, yyyy')}</div>
      <div className="font-mono font-bold text-lg" style={{ color: getRiskColor(score) }}>
        {formatScore(score)}
      </div>
      <div className="text-xs" style={{ color: getRiskColor(score) }}>{getRiskLabel(score)}</div>
    </div>
  )
}

export default function RiskScoreChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={COLOURS.riskExtreme} stopOpacity={0.3} />
            <stop offset="95%" stopColor={COLOURS.riskExtreme} stopOpacity={0}   />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={COLOURS.cardBorder} vertical={false} />
        <XAxis
          dataKey="date" tickFormatter={v => formatDate(v)}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, 100]} tickFormatter={v => v}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} width={35}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={40} stroke={COLOURS.riskElevated} strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: '40', fill: COLOURS.riskElevated, fontSize: 10, position: 'right' }} />
        <ReferenceLine y={60} stroke={COLOURS.riskHigh}     strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: '60', fill: COLOURS.riskHigh,     fontSize: 10, position: 'right' }} />
        <ReferenceLine y={75} stroke={COLOURS.riskExtreme}  strokeDasharray="4 4" strokeOpacity={0.5} label={{ value: '75', fill: COLOURS.riskExtreme,  fontSize: 10, position: 'right' }} />
        <Area
          type="monotone" dataKey="ensemble_score" name="Risk Score"
          stroke={COLOURS.riskExtreme} strokeWidth={1.5}
          fill="url(#riskGrad)" dot={false} activeDot={{ r: 4 }}
          animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
