import { motion } from 'framer-motion'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { Card } from '../ui/Card'
import clsx from 'clsx'

export default function KPICard({
  label, value, delta, deltaLabel,
  icon: Icon, valueColor, index = 0,
  loading = false,
}) {
  const isPos = delta > 0
  const isNeg = delta < 0

  if (loading) {
    return (
      <Card className="animate-pulse">
        <div className="h-3 w-20 bg-surface rounded mb-3" />
        <div className="h-8 w-28 bg-surface rounded mb-2" />
        <div className="h-3 w-14 bg-surface rounded" />
      </Card>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35 }}
    >
      <Card hover>
        <div className="flex items-start justify-between mb-3">
          <span className="text-text-secondary text-[11px] uppercase tracking-wider font-medium">
            {label}
          </span>
          {Icon && <Icon size={15} className="text-text-muted" />}
        </div>

        <div
          className="font-mono font-bold text-[28px] leading-none mb-2"
          style={{ color: valueColor || undefined, colorScheme: 'light' }}
        >
          {value ?? '—'}
        </div>

        {delta != null && (
          <div className={clsx(
            'flex items-center gap-1 text-xs font-mono',
            isPos ? 'text-risk-normal' : isNeg ? 'text-risk-extreme' : 'text-text-secondary'
          )}>
            {isPos ? <TrendingUp size={12} /> : isNeg ? <TrendingDown size={12} /> : <Minus size={12} />}
            <span>{isPos ? '+' : ''}{delta}</span>
            {deltaLabel && <span className="text-text-muted">{deltaLabel}</span>}
          </div>
        )}
      </Card>
    </motion.div>
  )
}
