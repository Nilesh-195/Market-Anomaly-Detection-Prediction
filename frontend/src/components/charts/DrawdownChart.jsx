import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { formatDate } from '../../utils/formatters'
import { COLOURS } from '../../constants/colours'

export default function DrawdownChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={COLOURS.riskExtreme} stopOpacity={0.3} />
            <stop offset="95%" stopColor={COLOURS.riskExtreme} stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={COLOURS.cardBorder} vertical={false} />
        <XAxis
          dataKey="date" tickFormatter={v => formatDate(v)}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} interval="preserveStartEnd"
        />
        <YAxis
          tickFormatter={v => `${(v * 100).toFixed(0)}%`}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} width={45}
        />
        <Tooltip
          contentStyle={{ background: COLOURS.cardBg, border: `1px solid ${COLOURS.cardBorder}`, borderRadius: 12 }}
          labelFormatter={v => formatDate(v, 'MMM dd, yyyy')}
          formatter={v => [`${(v * 100).toFixed(2)}%`, 'Drawdown']}
          labelStyle={{ color: COLOURS.textSecondary, fontSize: 11 }}
        />
        <ReferenceLine y={-0.1}  stroke={COLOURS.riskElevated} strokeDasharray="3 3" strokeOpacity={0.5} label={{ value: '-10%', fill: COLOURS.riskElevated, fontSize: 10, position: 'right' }} />
        <ReferenceLine y={-0.2}  stroke={COLOURS.riskHigh}     strokeDasharray="3 3" strokeOpacity={0.5} label={{ value: '-20%', fill: COLOURS.riskHigh,     fontSize: 10, position: 'right' }} />
        <ReferenceLine y={-0.3}  stroke={COLOURS.riskExtreme}  strokeDasharray="3 3" strokeOpacity={0.5} label={{ value: '-30%', fill: COLOURS.riskExtreme,  fontSize: 10, position: 'right' }} />
        <Area
          type="monotone" dataKey="drawdown" name="Drawdown"
          stroke={COLOURS.riskExtreme} strokeWidth={1.5}
          fill="url(#ddGrad)" dot={false}
          animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
