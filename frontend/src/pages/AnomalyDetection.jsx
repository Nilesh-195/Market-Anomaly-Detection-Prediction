import { 
  Activity, AlertTriangle, FastForward, Clock
} from 'lucide-react'
import { Card } from '../components/ui/Card'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, ReferenceLine } from 'recharts'
import { COLOURS } from '../constants/colours'
import { formatScore } from '../utils/formatters'
import { getRiskColor } from '../utils/riskHelpers'

// SVG Gauge Component for Ensemble Score
function ScoreGauge({ score }) {
  const radius = 60
  const circumference = radius * 2 * Math.PI
  const strokeDashoffset = circumference - (score / 100) * circumference

  return (
    <div className="relative flex items-center justify-center w-48 h-48 mx-auto">
      <svg className="transform -rotate-90 w-full h-full">
        {/* Background circle */}
        <circle
          cx="96" cy="96" r={radius}
          className="text-surface"
          strokeWidth="12"
          stroke="currentColor"
          fill="transparent"
        />
        {/* Progress circle */}
        <circle
          cx="96" cy="96" r={radius}
          stroke={getRiskColor(score)}
          strokeWidth="12"
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-mono font-bold" style={{ color: getRiskColor(score) }}>
          {formatScore(score)}
        </span>
        <span className="text-[10px] text-text-secondary uppercase tracking-widest mt-1">Ensemble</span>
      </div>
    </div>
  )
}

export default function AnomalyDetection({ current, evaluation, historical, loading }) {
  if (loading) {
    return <div className="h-64 bg-surface rounded-xl animate-pulse" />
  }

  const score = current?.ensemble_score ?? 0
  const breakdown = current?.model_scores ?? {}

  // Parse evaluation JSON if available for ROC-AUC chart
  const evalData = evaluation?.metrics ? Object.entries(evaluation.metrics).map(([model, m]) => ({
    name: model.toUpperCase(),
    auc: m.roc_auc,
    accuracy: m.accuracy
  })).sort((a, b) => (b.auc || 0) - (a.auc || 0)) : []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Anomaly Detection</h1>
        <p className="text-text-secondary text-sm">Deep learning crash radar and historical context mappings.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Gauge Hero */}
        <Card className="lg:col-span-1 flex flex-col items-center justify-center py-10">
          <h2 className="text-lg font-semibold text-text-primary mb-6">Current Market Threat</h2>
          <ScoreGauge score={score} />
          <div className="mt-8 w-full space-y-4 px-4">
            {Object.entries(breakdown).map(([model, s]) => (
              <div key={model}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-text-secondary uppercase font-mono">{model}</span>
                  <span className="font-mono font-bold" style={{ color: getRiskColor(s) }}>{formatScore(s)}</span>
                </div>
                <div className="h-1.5 w-full bg-surface rounded-full overflow-hidden">
                  <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${s}%`, backgroundColor: getRiskColor(s) }} />
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Global Evaluation Metrics */}
        <Card className="lg:col-span-2">
          <div className="flex items-center gap-2 mb-6">
             <Activity size={20} className="text-brand-blue" />
             <h2 className="text-lg font-semibold text-text-primary">Global Model Performance (ROC-AUC)</h2>
          </div>
          {evalData.length > 0 ? (
            <div className="w-full h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={evalData} margin={{ top: 10, right: 10, bottom: 0, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={COLOURS.cardBorder} />
                  <XAxis dataKey="name" tick={{ fontSize: 11, fill: COLOURS.textSecondary, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: COLOURS.textSecondary, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                  <Tooltip
                    cursor={{ fill: 'transparent' }}
                    content={({ active, payload }) => {
                      if (!active || !payload?.length) return null
                      const d = payload[0].payload
                      return (
                        <div className="bg-card-bg border border-card-border p-3 rounded-lg shadow-lg text-sm">
                          <p className="font-mono font-bold text-text-primary mb-1">{d.name}</p>
                          <div className="flex justify-between gap-4"><span className="text-text-secondary">ROC-AUC</span><span className="font-mono">{d.auc?.toFixed(3)}</span></div>
                          <div className="flex justify-between gap-4"><span className="text-text-secondary">Accuracy</span><span className="font-mono">{d.accuracy?.toFixed(3)}</span></div>
                        </div>
                      )
                    }}
                  />
                  <Bar dataKey="auc" radius={[4, 4, 0, 0]} barSize={40}>
                    {evalData.map((d, i) => <Cell key={i} fill={COLOURS.brandBlue} opacity={0.6 + (d.auc * 0.4)} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-text-muted text-sm">Global evaluation data not available</div>
          )}
        </Card>
      </div>

    </div>
  )
}
