import { useMemo } from 'react'
import { formatDate } from '../../utils/formatters'
import { getRiskColor } from '../../utils/riskHelpers'
import clsx from 'clsx'

export default function AnomalyCalendar({ events = [] }) {
  // Group events by year and month
  const calendarData = useMemo(() => {
    const grouped = {}

    events.forEach(ev => {
      if (!ev.date) return
      const date = new Date(ev.date)
      const year = date.getFullYear()
      const month = date.getMonth()
      const day = date.getDate()

      if (!grouped[year]) grouped[year] = {}
      if (!grouped[year][month]) grouped[year][month] = {}
      grouped[year][month][day] = ev.ensemble_score ?? 0
    })

    return grouped
  }, [events])

  const years = Object.keys(calendarData).sort((a, b) => b - a).slice(0, 3)
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

  if (events.length === 0) {
    return (
      <div className="flex items-center justify-center py-10 text-text-muted">
        No anomaly events to display
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {years.map(year => (
        <div key={year}>
          <div className="text-sm font-semibold text-text-primary mb-3">{year}</div>
          <div className="grid grid-cols-12 gap-1">
            {months.map((month, monthIdx) => {
              const monthData = calendarData[year]?.[monthIdx] || {}
              const daysInMonth = new Date(parseInt(year), monthIdx + 1, 0).getDate()
              const hasAnomalies = Object.keys(monthData).length > 0

              return (
                <div key={monthIdx} className="flex flex-col items-center">
                  <div className="text-[10px] text-text-muted mb-1">{month}</div>
                  <div className="flex flex-wrap gap-0.5 w-6 justify-center">
                    {Array.from({ length: Math.min(daysInMonth, 31) }).map((_, day) => {
                      const score = monthData[day + 1]
                      const hasAnomaly = score !== undefined
                      return (
                        <div
                          key={day}
                          className={clsx(
                            'w-1 h-1 rounded-full transition-all',
                            hasAnomaly ? '' : 'bg-surface'
                          )}
                          style={hasAnomaly ? { backgroundColor: getRiskColor(score) } : undefined}
                          title={hasAnomaly ? `${year}-${String(monthIdx + 1).padStart(2, '0')}-${String(day + 1).padStart(2, '0')}: Score ${score.toFixed(1)}` : undefined}
                        />
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 pt-4 border-t border-card-border">
        <div className="flex items-center gap-1 text-xs text-text-muted">
          <span className="w-2 h-2 rounded-full bg-risk-normal"></span>
          Normal
        </div>
        <div className="flex items-center gap-1 text-xs text-text-muted">
          <span className="w-2 h-2 rounded-full bg-risk-elevated"></span>
          Elevated
        </div>
        <div className="flex items-center gap-1 text-xs text-text-muted">
          <span className="w-2 h-2 rounded-full bg-risk-high"></span>
          High
        </div>
        <div className="flex items-center gap-1 text-xs text-text-muted">
          <span className="w-2 h-2 rounded-full bg-risk-extreme"></span>
          Extreme
        </div>
      </div>
    </div>
  )
}
