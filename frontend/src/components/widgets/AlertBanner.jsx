import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, X } from 'lucide-react'
import { useState } from 'react'
import { Badge } from '../ui/Badge'
import { getRiskLabel } from '../../utils/riskHelpers'
import { formatScore } from '../../utils/formatters'
import { RISK_THRESHOLDS } from '../../constants/config'

export default function AlertBanner({ current }) {
  const [dismissed, setDismissed] = useState(false)
  const score = current?.ensemble_score ?? 0
  const show = score >= RISK_THRESHOLDS.elevated && !dismissed

  const flagged = []
  const s = current?.model_scores || {}
  if ((s.zscore  ?? 0) >= RISK_THRESHOLDS.elevated) flagged.push('Z-SCORE')
  if ((s.iforest ?? 0) >= RISK_THRESHOLDS.elevated) flagged.push('ISO FOREST')
  if ((s.lstm    ?? 0) >= RISK_THRESHOLDS.elevated) flagged.push('LSTM')
  if ((s.prophet ?? 0) >= RISK_THRESHOLDS.elevated) flagged.push('PROPHET')

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, height: 0, y: -10 }}
          animate={{ opacity: 1, height: 'auto', y: 0 }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.25 }}
          className="overflow-hidden"
        >
          <div className="flex items-center gap-4 bg-[#1a0a0a] border-l-4 border-risk-extreme rounded-xl p-4 mb-4">
            <AlertTriangle
              size={20}
              className="text-risk-extreme flex-shrink-0 animate-pulse"
            />
            <div className="flex-1 flex flex-wrap items-center gap-3">
              <span className="text-[#F1F5F9] font-semibold text-sm">
                ANOMALY DETECTED
              </span>
              <div className="flex items-center gap-1.5 flex-wrap">
                {flagged.map(f => (
                  <Badge key={f} variant="red">{f}</Badge>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="font-mono font-bold text-xl text-risk-extreme">
                  {formatScore(score)}
                </div>
                <div className="text-xs text-risk-extreme">{getRiskLabel(score)}</div>
              </div>
              <button
                onClick={() => setDismissed(true)}
                className="text-[#64748B] hover:text-[#F1F5F9] transition-colors"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
