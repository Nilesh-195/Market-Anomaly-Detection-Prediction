import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell
} from 'recharts'
import { format } from 'date-fns'


const getBarColor = (score) => {
  if (score >= 75) return '#EF4444' // red
  if (score >= 60) return '#F97316' // orange
  if (score >= 40) return '#F59E0B' // amber
  return '#10B981' // green
}

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const score = payload[0].value
    return (
      <div className="rounded-lg border border-card-border bg-card-bg p-3 shadow-lg">
        <p className="mb-1 text-xs font-semibold text-text-secondary">{format(new Date(label), 'MMM d, yyyy')}</p>
        <p className="font-mono text-sm font-bold text-text-primary">
          Score: <span style={{ color: getBarColor(score) }}>{score.toFixed(1)}</span>
        </p>
        <p className="mt-1 text-[10px] text-text-muted">Forecasted anomaly score (not price)</p>
      </div>
    )
  }
  return null
}

export default function AnomalyForecastChart({ forecastData }) {
  if (!forecastData || !forecastData.dates || !forecastData.values) {
    return (
      <div className="h-[220px] flex items-center justify-center rounded-lg border border-dashed border-card-border bg-surface/50 text-text-muted text-sm px-4 text-center">
        No forecast data available for this horizon.
      </div>
    )
  }

  const { dates, values } = forecastData
  
  const data = dates.map((dateStr, i) => ({
    date: dateStr,
    displayDate: format(new Date(dateStr), 'MMM d'),
    score: values[i] ?? 0
  }))



  return (
    <div className="h-[250px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#EAEBEE" />
          <XAxis 
            dataKey="date" 
            tickFormatter={(val) => format(new Date(val), 'MMM d')}
            tick={{ fontSize: 11, fill: '#6B7280' }} 
            axisLine={false} 
            tickLine={false} 
            dy={10}
          />
          <YAxis 
            tick={{ fontSize: 11, fill: '#6B7280' }} 
            axisLine={false} 
            tickLine={false} 
            domain={[0, 100]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: '#F0F2F5' }} />
          <ReferenceLine y={60} stroke="#F59E0B" strokeDasharray="4 4" label={{ position: 'top', value: 'Threshold (60)', fill: '#F59E0B', fontSize: 10 }} />
          <Bar dataKey="score" radius={[4, 4, 0, 0]} maxBarSize={40}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getBarColor(entry.score)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
