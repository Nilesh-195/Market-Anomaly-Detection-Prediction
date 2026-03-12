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
  // Build synthetic ROC points from AUC values
  if (!evaluation) return null

  const models = Object.entries(evaluation).map(([asset, data]) => ({
    asset,
    auc: data?.auc_score,
  }))

  // Generate approximate ROC curve points from AUC
  function rocPoints(auc = 0.5) {
    const pts = []
    for (let i = 0; i <= 20; i++) {
      const fpr = i / 20
      const tpr = auc < 0.5
        ? fpr
        : Math.min(1, fpr + 2 * (auc - 0.5) * Math.sqrt(fpr * (1 - fpr) + 0.01))
      pts.push({ fpr, tpr })
    }
    return pts
  }

  // Merge all model points
  const allPoints = rocPoints(0.5).map((_, idx) => {
    const obj = { fpr: idx / 20 }
    models.forEach(m => {
      const pts = rocPoints(m.auc)
      obj[m.asset] = pts[idx]?.tpr
    })
    return obj
  })

  const modelColors = [COLOURS.chartBlue, COLOURS.chartPurple, COLOURS.chartCyan, COLOURS.chartGreen, COLOURS.riskHigh, COLOURS.riskElevated]

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
        {/* Diagonal */}
        <ReferenceLine segment={[{x:0,y:0},{x:1,y:1}]} stroke={COLOURS.textMuted} strokeDasharray="4 4" />
        {models.map((m, i) => (
          <Line
            key={m.asset}
            dataKey={m.asset}
            name={`${m.asset} (AUC ${m.auc?.toFixed(3)})`}
            stroke={modelColors[i % modelColors.length]}
            strokeWidth={1.5} dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
