import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { COLOURS } from '../../constants/colours'
import { formatDate } from '../../utils/formatters'
import { getRiskColor } from '../../utils/riskHelpers'

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const val = payload[0].value
    return (
      <div className="bg-card-bg border border-card-border px-3 py-2 rounded-xl shadow-lg">
        <p className="font-mono text-[10px] text-text-secondary uppercase tracking-wider mb-1">{formatDate(label, 'MMM dd, yyyy')}</p>
        <div className="flex items-center gap-3">
           <span className="text-xs text-text-secondary">Risk Score</span>
           <span className="font-mono font-bold text-[14px]" style={{ color: getRiskColor(val) }}>
             {val.toFixed(1)}
           </span>
        </div>
      </div>
    )
  }
  return null
}

export default function RiskScoreChart({ data }) {
  if (!data || data.length === 0) return null
  return (
    <div style={{ width: '100%', height: 220 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 0, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLOURS.cardBorder} opacity={0.6} />
          <XAxis
            dataKey="date"
            axisLine={false}
            tickLine={false}
            tick={{ fill: COLOURS.textMuted, fontSize: 10, fontFamily: 'monospace' }}
            tickFormatter={(val) => formatDate(val, 'MMM dd')}
            minTickGap={40}
          />
          <YAxis
            domain={[0, 100]}
            axisLine={false}
            tickLine={false}
            tick={{ fill: COLOURS.textMuted, fontSize: 10, fontFamily: 'monospace' }}
            width={30}
            orientation="right"
          />
          <ReferenceLine y={60} stroke="#F59E0B" strokeDasharray="3 3" opacity={0.5} strokeWidth={1.5} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: COLOURS.surface, opacity: 0.4 }} />
          <Line
            type="monotone"
            dataKey="ensemble_score"
            stroke={COLOURS.riskHigh}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, stroke: '#fff', fill: COLOURS.riskHigh }}
            isAnimationActive={true}
            animationDuration={800}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
