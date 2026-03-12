import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { getRiskColor, getRiskLabel, getRiskBgTailwind } from '../../utils/riskHelpers'
import { formatScore, formatDate } from '../../utils/formatters'

export default function ForecastDayCard({ day, date, score, delta, index = 0 }) {
  const color = getRiskColor(score)
  const label = getRiskLabel(score)

  const badgeVariant = score < 40 ? 'green' : score < 60 ? 'yellow' : score < 75 ? 'orange' : 'red'

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07, duration: 0.3 }}
    >
      <Card hover className="text-center">
        <div className="text-[#64748B] text-[11px] uppercase tracking-wider mb-1">{day}</div>
        <div className="text-[#334155] text-xs font-mono mb-3">{formatDate(date, 'MMM dd')}</div>

        <div className="font-mono font-bold text-3xl leading-none mb-1" style={{ color }}>
          {formatScore(score)}
        </div>

        {delta != null && (
          <div className={`flex items-center justify-center gap-1 text-xs font-mono mb-3 ${delta >= 0 ? 'text-risk-extreme' : 'text-risk-normal'}`}>
            {delta >= 0 ? <TrendingUp size={11} /> : <TrendingDown size={11} />}
            <span>{delta >= 0 ? '+' : ''}{formatScore(delta)}</span>
          </div>
        )}

        {/* Risk bar */}
        <div className="w-full h-1.5 bg-surface rounded-full overflow-hidden mb-2">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${Math.min(score, 100)}%`, background: color }}
          />
        </div>

        <Badge variant={badgeVariant}>{label}</Badge>
      </Card>
    </motion.div>
  )
}
