import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { getRiskLabel } from '../../utils/riskHelpers'
import { formatScore } from '../../utils/formatters'
import { MODEL_COLOURS } from '../../constants/colours'

const MODELS = [
  { key: 'zscore',  label: 'Z-SCORE',          variant: 'blue',   weight: 15 },
  { key: 'iforest', label: 'ISOLATION FOREST',  variant: 'purple', weight: 25 },
  { key: 'lstm',    label: 'LSTM AUTOENCODER',  variant: 'cyan',   weight: 40 },
  { key: 'prophet', label: 'PROPHET',           variant: 'green',  weight: 20 },
]

function DotBar({ score, color }) {
  const filled = Math.round((score / 100) * 5)
  return (
    <div className="flex gap-1">
      {Array.from({ length: 5 }).map((_, i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full transition-all duration-500"
          style={{ background: i < filled ? color : '#D1D5DB' }}
        />
      ))}
    </div>
  )
}

export default function ModelConsensus({ current }) {
  const scores = current?.model_scores || {}
  const ensemble = current?.ensemble_score ?? 0

  return (
    <Card>
      <div className="text-text-primary font-medium mb-0.5">Model Consensus</div>
      <div className="text-text-secondary text-xs mb-4">Current detection status</div>

      <div className="space-y-3">
        {MODELS.map(m => {
          const score = scores[m.key] ?? 0
          const label = getRiskLabel(score)
          const badgeVariant = score < 40 ? 'green' : score < 60 ? 'yellow' : score < 75 ? 'orange' : 'red'
          return (
            <div key={m.key} className="flex items-center gap-3">
              <Badge variant={m.variant} className="w-[140px] justify-center text-[10px]">
                {m.label}
              </Badge>
              <DotBar score={score} color={MODEL_COLOURS[m.key]} />
              <Badge variant={badgeVariant} className="ml-auto text-[10px]">{label}</Badge>
            </div>
          )
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-card-border">
        <div className="text-text-secondary text-[11px] uppercase tracking-wider mb-2">Ensemble Score</div>
        <div className="text-text-muted text-xs font-mono mb-1">
          15% + 25% + 40% + 20% =
        </div>
        <div
          className="font-mono font-bold text-2xl"
          style={{ color: MODEL_COLOURS.lstm }}
        >
          {formatScore(ensemble)}<span className="text-sm text-text-secondary">/100</span>
        </div>
      </div>
    </Card>
  )
}
