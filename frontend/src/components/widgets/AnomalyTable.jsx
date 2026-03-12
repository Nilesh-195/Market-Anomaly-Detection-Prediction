import { motion } from 'framer-motion'
import { Badge } from '../ui/Badge'
import { SkeletonRow } from '../ui/Skeleton'
import { formatDate, formatScore } from '../../utils/formatters'
import { AlertCircle } from 'lucide-react'

export default function AnomalyTable({ events = [], loading = false, maxRows = 8 }) {
  const rows = events.slice(0, maxRows)

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-card-border">
            {['Date', 'Ensemble Score', 'Risk Level', 'Severity'].map(h => (
              <th key={h} className="text-left text-[#64748B] text-[11px] uppercase tracking-wider pb-2 px-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading
            ? Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}><td colSpan={4}><SkeletonRow /></td></tr>
              ))
            : rows.length === 0
            ? (
              <tr>
                <td colSpan={4}>
                  <div className="flex flex-col items-center justify-center py-10 text-[#334155]">
                    <AlertCircle size={32} className="mb-2 opacity-40" />
                    <div className="text-sm">No anomalies detected in this period</div>
                  </div>
                </td>
              </tr>
            )
            : rows.map((ev, i) => {
                const score = ev.ensemble_score ?? 0
                const label = ev.risk_label ?? '—'
                const badgeVariant = score < 40 ? 'green' : score < 60 ? 'yellow' : score < 75 ? 'orange' : 'red'
                const barPct = Math.min(score, 100)
                const barColor = score < 40 ? '#22c55e' : score < 60 ? '#eab308' : score < 75 ? '#f97316' : '#ef4444'
                return (
                  <motion.tr
                    key={i}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.04 }}
                    className="border-b border-card-border/50 hover:bg-surface transition-colors"
                  >
                    <td className="py-2.5 px-2 font-mono text-[#64748B] text-xs">
                      {formatDate(ev.date, 'MMM dd, yyyy')}
                    </td>
                    <td className="py-2.5 px-2 font-mono font-semibold text-sm" style={{ color: barColor }}>
                      {formatScore(score)}
                    </td>
                    <td className="py-2.5 px-2">
                      <Badge variant={badgeVariant}>{label}</Badge>
                    </td>
                    <td className="py-2.5 px-2 w-28">
                      <div className="h-1.5 bg-surface rounded-full overflow-hidden w-24">
                        <div className="h-full rounded-full" style={{ width: `${barPct}%`, background: barColor }} />
                      </div>
                    </td>
                  </motion.tr>
                )

              })
          }
        </tbody>
      </table>
    </div>
  )
}
