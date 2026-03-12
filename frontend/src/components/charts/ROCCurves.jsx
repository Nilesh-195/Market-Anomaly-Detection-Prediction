import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'
import { COLOURS, MODEL_COLOURS } from '../../constants/colours'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-3 shadow-2xl min-w-[200px]">
      <div className="text-[#64748B] text-xs mb-2">False Positive Rate</div>
      {payload.map((p, i) => (
        <div key={i} className="flex justify-between gap-4 text-xs">
          <span style={{ color: p.color }}>{p.name}</span>
          <span className="font-mono" style={{ color: p.color }}>
            TPR: {p.value?.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  )
}

export default function ROCCurves({ evaluation }) {
  if (!evaluation) return null

  const MODEL_KEYS = ['zscore_score', 'iforest_score', 'lstm_score', 'prophet_score']
  const MODEL_DISPLAY = {
    zscore_score:  'Z-Score',
    iforest_score: 'Iso Forest',
    lstm_score:    'LSTM',
    prophet_score: 'Prophet',
  }

  // Average AUC for each model across all assets
  const models = MODEL_KEYS.map(mk => {
    const aucs = Object.values(evaluation)
      .map(asset => asset?.[mk]?.roc_auc)
      .filter(v => v != null)
    const auc = aucs.length ? aucs.reduce((s, v) => s + v, 0) / aucs.length : 0.5
    return { key: mk, label: MODEL_DISPLAY[mk], auc }
  })

  // Generate approximate ROC curve from AUC
  function rocPoints(auc = 0.5) {
    const pts = []
    for (let i = 0; i <= 20; i++) {
      const fpr = i / 20
      const tpr = Math.min(1, fpr + 2 * (auc - 0.5) * Math.sqrt(fpr * (1 - fpr) + 0.01))
      pts.push({ fpr, tpr })
    }
    return pts
  }

  const allPoints = rocPoints(0.5).map((_, idx) => {
    const obj = { fpr: idx / 20 }
    models.forEach(m => {
      obj[m.key] = rocPoints(m.auc)[idx]?.tpr
    })
    return obj
  })

  const modelColors = [COLOURS.chartBlue, COLOURS.chartPurple, COLOURS.chartCyan, COLOURS.chartGreen]

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={allPoints} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={COLOURS.cardBorder} />
        <XAxis
          dataKey="fpr" tickFormatter={v => v.toFixed(1)}
          label={{ value: 'False Positive Rate', fill: COLOURS.textSecondary, fontSize: 11, position: 'insideBottom', offset: -2 }}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          label={{ value: 'True Positive Rate', fill: COLOURS.textSecondary, fontSize: 11, angle: -90, position: 'insideLeft' }}
          tick={{ fill: COLOURS.textSecondary, fontSize: 11, fontFamily: 'monospace' }}
          axisLine={false} tickLine={false} width={35}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 11, color: COLOURS.textSecondary }} />
        <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke={COLOURS.textMuted} strokeDasharray="4 4" />
        {models.map((m, i) => (
          <Line
            key={m.key}
            dataKey={m.key}
            name={`${m.label} (AUC ${m.auc.toFixed(3)})`}
            stroke={modelColors[i]}
            strokeWidth={2} dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
