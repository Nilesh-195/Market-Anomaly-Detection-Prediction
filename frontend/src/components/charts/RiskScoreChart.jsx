import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import { COLOURS } from '../../constants/colours'
import { formatDate } from '../../utils/formatters'

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const val = payload[0].value
    return (
      <div className="bg-white border border-slate-200 px-3 py-2 rounded-xl shadow-panel">
        <p className="font-mono text-[10px] text-slate-500 mb-1">{formatDate(label)}</p>
        <p className="font-mono font-bold text-[14px]" style={{ color: payload[0].color }}>
          {val.toFixed(1)}
        </p>
      </div>
    )
  }
  return null
}

export default function RiskScoreChart({ data }) {
  if (!data || data.length === 0) return null
  return (
    <div style={{ width: '100%', height: 200 }}>
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
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: COLOURS.borderBright, strokeWidth: 1, strokeDasharray: '4 4' }} />
          <Line
            type="monotone"
            dataKey="ensemble_score"
            stroke={COLOURS.riskHigh}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0, fill: COLOURS.riskHigh }}
            isAnimationActive={true}
            animationDuration={800}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
