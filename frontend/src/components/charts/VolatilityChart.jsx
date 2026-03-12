import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from 'recharts'
import { formatDate } from '../../utils/formatters'
import { COLOURS } from '../../constants/colours'

export default function VolatilityChart({ data = [] }) {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={COLOURS.riskElevated} stopOpacity={0.3} />
            <stop offset="95%" stopColor={COLOURS.riskElevated} stopOpacity={0}   />
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
          axisLine={false} tickLine={false} width={40}
        />
        <Tooltip
          contentStyle={{ background: COLOURS.cardBg, border: `1px solid ${COLOURS.cardBorder}`, borderRadius: 12 }}
          labelStyle={{ color: COLOURS.textSecondary, fontSize: 11 }}
          labelFormatter={v => formatDate(v, 'MMM dd, yyyy')}
          formatter={v => [`${(v * 100).toFixed(1)}%`, 'Volatility']}
        />
        <Area
          type="monotone" dataKey="vol_30" name="Volatility"
          stroke={COLOURS.riskElevated} strokeWidth={1.5}
          fill="url(#volGrad)" dot={false}
          animationDuration={800}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
