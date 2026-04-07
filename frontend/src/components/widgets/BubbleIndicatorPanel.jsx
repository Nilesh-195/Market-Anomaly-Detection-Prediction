import { useMemo } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid } from 'recharts'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { formatDate } from '../../utils/formatters'

function getBubbleLevel(score) {
  if (score < 10) return { label: 'Normal', variant: 'green', color: '#15803D' }
  if (score <= 25) return { label: 'Overextended', variant: 'yellow', color: '#B45309' }
  return { label: 'Extreme', variant: 'red', color: '#B91C1C' }
}

function BubbleTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const value = Number(payload[0]?.value)
  return (
    <div className="rounded-lg border border-card-border bg-white px-3 py-2 shadow-float">
      <div className="text-xs font-semibold uppercase tracking-[0.1em] text-text-muted">{formatDate(label, 'MMM dd, yyyy')}</div>
      <div className="mt-1 font-mono text-sm font-bold text-text-primary">{Number.isFinite(value) ? `${value.toFixed(2)}%` : '—'}</div>
    </div>
  )
}

export default function BubbleIndicatorPanel({ historyChartData = [], currentBubbleScore = null }) {
  const bubbleSeries = useMemo(
    () => (historyChartData ?? [])
      .filter((row) => Number.isFinite(Number(row?.bubble_score)))
      .map((row) => ({ date: row.date, bubble_score: Number(row.bubble_score) })),
    [historyChartData]
  )

  const recent = bubbleSeries.slice(-90)
  const derivedCurrent = recent.length ? recent[recent.length - 1].bubble_score : null
  const currentScore = Number.isFinite(Number(currentBubbleScore)) ? Number(currentBubbleScore) : derivedCurrent
  const level = getBubbleLevel(Number.isFinite(currentScore) ? currentScore : 0)

  return (
    <Card className="p-4">
      <h3 className="text-sm font-bold uppercase tracking-[0.12em] text-text-primary">Bubble / Overextension</h3>
      <p className="mt-1 text-xs text-text-secondary">bubble_score = % distance from 200-day average (proxy for overextension)</p>

      <div className="mt-3 flex items-center justify-between">
        <div className="font-mono text-3xl font-bold" style={{ color: level.color }}>
          {Number.isFinite(currentScore) ? `${currentScore.toFixed(2)}%` : '—'}
        </div>
        <Badge variant={level.variant}>{level.label}</Badge>
      </div>

      {!recent.length ? (
        <div className="mt-3 rounded-lg border border-dashed border-card-border bg-surface/40 px-4 py-6 text-center text-sm text-text-muted">
          Bubble score history unavailable.
        </div>
      ) : (
        <div className="mt-3 h-[170px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={recent} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="#DBE4EF" strokeDasharray="4 4" vertical={false} />
              <XAxis dataKey="date" tickFormatter={(value) => formatDate(value)} tick={{ fill: '#7C8BA1', fontSize: 11 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: '#7C8BA1', fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(0)}%`} tickLine={false} axisLine={false} width={42} />
              <Tooltip content={<BubbleTooltip />} />
              <Line type="monotone" dataKey="bubble_score" stroke="#1D6FDC" strokeWidth={2.2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}
