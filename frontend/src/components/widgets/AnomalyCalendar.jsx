import { formatDate, formatScore } from '../../utils/formatters'
import { getRiskColor } from '../../utils/riskHelpers'

export default function AnomalyCalendar({ events = [] }) {
  // Build a map of date → score
  const scoreMap = {}
  events.forEach(ev => {
    const d = String(ev.date).slice(0, 10)
    scoreMap[d] = Math.max(scoreMap[d] ?? 0, ev.ensemble_score ?? ev.score ?? 0)
  })

  // Generate last 2 years of dates
  const today = new Date()
  const start = new Date(today)
  start.setFullYear(today.getFullYear() - 2)

  const days = []
  const cur = new Date(start)
  while (cur <= today) {
    days.push(new Date(cur))
    cur.setDate(cur.getDate() + 1)
  }

  // Group by month
  const months = {}
  days.forEach(d => {
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    if (!months[key]) months[key] = []
    months[key].push(d)
  })

  function scoreToColor(score) {
    if (!score) return '#1A2640'
    if (score < 40)  return `rgba(16,185,129,${score / 100})`
    if (score < 60)  return `rgba(245,158,11,${score / 100})`
    if (score < 75)  return `rgba(249,115,22,${score / 100})`
    return `rgba(239,68,68,${0.3 + score / 200})`
  }

  return (
    <div className="overflow-x-auto">
      <div className="flex gap-2 min-w-max">
        {Object.entries(months).map(([monthKey, monthDays]) => {
          const [year, month] = monthKey.split('-')
          const label = new Date(+year, +month - 1).toLocaleString('en-US', { month: 'short', year: '2-digit' })
          // Pad to start on correct weekday
          const firstDay = monthDays[0].getDay()
          return (
            <div key={monthKey} className="flex flex-col gap-1">
              <div className="text-[#334155] text-[10px] font-mono mb-1">{label}</div>
              <div className="grid grid-cols-7 gap-[3px]">
                {Array.from({ length: firstDay }).map((_, i) => (
                  <div key={`pad-${i}`} className="w-[10px] h-[10px]" />
                ))}
                {monthDays.map(d => {
                  const key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`
                  const score = scoreMap[key]
                  return (
                    <div
                      key={key}
                      title={score ? `${formatDate(key, 'MMM dd, yyyy')}: ${formatScore(score)}` : formatDate(key, 'MMM dd')}
                      className="w-[10px] h-[10px] rounded-[2px] cursor-pointer hover:ring-1 hover:ring-white/20 transition-all"
                      style={{ background: scoreToColor(score) }}
                    />
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
      <div className="flex items-center gap-3 mt-3">
        <span className="text-[#64748B] text-[11px]">Less</span>
        {[0, 30, 50, 65, 85].map(s => (
          <div key={s} className="w-3 h-3 rounded-[2px]" style={{ background: scoreToColor(s || 0) }} />
        ))}
        <span className="text-[#64748B] text-[11px]">More anomalous</span>
      </div>
    </div>
  )
}
