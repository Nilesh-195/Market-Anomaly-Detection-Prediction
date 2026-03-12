import { motion } from 'framer-motion'
import { Badge } from '../ui/Badge'
import { SkeletonRow } from '../ui/Skeleton'
import { formatDate, formatScore, formatZScore } from '../../utils/formatters'
import { getRiskBgTailwind } from '../../utils/riskHelpers'
import { AlertCircle } from 'lucide-react'

const MODEL_KEYS = ['zscore', 'iforest', 'lstm', 'prophet']
const MODEL_LABELS = { zscore: 'Z', iforest: 'IF', lstm: 'LSTM', prophet: 'PRO' }
const MODEL_VARIANTS = { zscore: 'blue', iforest: 'purple', lstm: 'cyan', prophet: 'green' }

export default function AnomalyTable({ events = [], loading = false, maxRows = 8 }) {
  const rows = events.slice(0, maxRows)

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-card-border">
            {['Date', 'Risk Score', 'Z-Score', 'Models Flagged', 'Status'].map(h => (
              <th key={h} className="text-left text-[#64748B] text-[11px] uppercase tracking-wider pb-2 px-2 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading
            ? Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}><td colSpan={5}><SkeletonRow /></td></tr>
              ))
            : rows.length === 0
            ? (
              <tr>
                <td colSpan={5}>
                  <div className="flex flex-col items-center justify-center py-10 text-[#334155]">
                    <AlertCircle size={32} className="mb-2 opacity-40" />
                    <div className="text-sm">No anomalies detected in this period</div>
                  </div>
                </td>
              </tr>
            )
            : rows.map((ev, i) => {
                const score = ev.ensemble_score ?? ev.score ?? 0
                const badgeVariant = score < 40 ? 'green' : score < 60 ? 'yellow' : score < 75 ? 'orange' : 'red'
                const flagged = MODEL_KEYS.filter(k => (ev.model_scores?.[k] ?? 0) >= 40)
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
                    <td className="py-2.5 px-2">
                      <Badge variant={badgeVariant}>{formatScore(score)}</Badge>
                    </td>
                    <td className="py-2.5 px-2 font-mono text-xs text-[#64748B]">
                      {ev.zscore_val != null ? formatZScore(ev.zscore_val) : '—'}
                    </td>
                    <td className="py-2.5 px-2">
                      <div className="flex gap-1 flex-wrap">
                        {flagged.length > 0
                          ? flagged.map(k => (
                              <Badge key={k} variant={MODEL_VARIANTS[k]}>{MODEL_LABELS[k]}</Badge>
                            ))
                          : <span className="text-[#334155] text-xs">—</span>
                        }
                      </div>
                    </td>
                    <td className="py-2.5 px-2">
                      <Badge variant={ev.confirmed ? 'red' : 'default'}>
                        {ev.confirmed ? 'CONFIRMED' : 'FLAGGED'}
                      </Badge>
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
