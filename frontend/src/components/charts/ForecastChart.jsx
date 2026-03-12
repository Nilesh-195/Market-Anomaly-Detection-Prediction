import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { formatDate, formatScore } from '../../utils/formatters'
import { COLOURS } from '../../constants/colours'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-3 shadow-2xl min-w-[180px]">
      <div className="text-[#64748B] text-xs mb-2">{formatDate(label, 'MMM dd, yyyy')}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex justify-between gap-4 text-xs">
          <span className="text-[#64748B]">{p.name}</span>
          <span className="font-mono font-medium" style={{ color: p.color }}>
            {p.name?.includes('CI') ? formatScore(p.value[1]) : formatScore(p.value)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function ForecastChart({ historical = [], forecast = [] }) {
  // Mark the boundary
  const boundary = historical[historical.length - 1]?.date

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart
        data={[...historical, ...forecast]}
        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={COLOURS.chartBlue} stopOpacity={0.2} />
            <stop offset="95%" stopColor={COLOURS.chartBlue} stopOpacity={0}   />
          </linearGradient>
          <linearGradient id="foreGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={COLOURS.chartCyan} stopOpacity={0.2} />
            <stop offset="95%" stopColor={COLOURS.chartCyan} stopOpacity={0}   />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke={COLOURS.cardBorder} vertical={false} />
        <XAxis
          dataKey="date" tickFormatter={v => formatDate(v)}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} width={35}
        />
        <Tooltip content={<CustomTooltip />} />
        {boundary && (
          <ReferenceLine
            x={boundary}
            stroke={COLOURS.textSecondary}
            strokeDasharray="4 4"
            label={{ value: 'TODAY', fill: COLOURS.textSecondary, fontSize: 10, position: 'insideTopLeft' }}
          />
        )}
        <ReferenceLine y={40} stroke={COLOURS.riskElevated} strokeDasharray="3 3" strokeOpacity={0.4} />
        <ReferenceLine y={60} stroke={COLOURS.riskHigh}     strokeDasharray="3 3" strokeOpacity={0.4} />

        {/* Confidence band */}
        <Area
          dataKey="ci" name="CI"
          stroke="none" fill={COLOURS.chartCyan}
          fillOpacity={0.08} dot={false}
          connectNulls={false}
        />

        {/* Historical line */}
        <Line
          dataKey="ensemble_score" name="Historical"
          stroke={COLOURS.chartBlue} strokeWidth={1.5}
          dot={false} activeDot={{ r: 4 }}
          connectNulls={false}
        />

        {/* Forecast line */}
        <Line
          dataKey="forecast_score" name="Forecast"
          stroke={COLOURS.chartCyan} strokeWidth={1.5}
          strokeDasharray="5 3"
          dot={false} activeDot={{ r: 4 }}
          connectNulls={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
