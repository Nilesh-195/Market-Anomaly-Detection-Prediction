import { BarChart as BarIcon, Trophy, Table } from 'lucide-react'
import { Card } from '../components/ui/Card'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts'
import { COLOURS } from '../constants/colours'

export default function MethodCompare({ comparison, loading, error }) {
  if (loading) {
    return <div className="h-64 flex items-center justify-center animate-pulse bg-surface rounded-xl" />
  }

  if (error || !comparison || !comparison.models) {
    return (
      <Card className="bg-red-50 border-red-200 text-red-600 p-6 flex flex-col items-center">
        <span>Failed to load method comparison. Data may be unavailable.</span>
      </Card>
    )
  }

  // Model comparison format from backend typically:
  // comparison.models = { arima: { rmse: 10.5, mae: 8.2, mape: 1.5 }, prophet: { ... }, ... }
  // Transform to array and sort by RMSE
  const modelsData = Object.entries(comparison.models).map(([key, metrics]) => ({
    name: key.toUpperCase(),
    rmse: metrics.rmse,
    mae: metrics.mae,
    mape: metrics.mape,
  })).sort((a, b) => (a.rmse || 0) - (b.rmse || 0))

  const bestModel = modelsData[0]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">Method Comparison</h1>
        <p className="text-text-secondary text-sm">Compare prediction accuracy across statistical and deep learning models.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Ranked Horizontal Bar Chart */}
        <Card className="flex flex-col h-[400px]">
          <div className="flex items-center gap-2 mb-6">
            <BarIcon size={20} className="text-brand-blue" />
            <h2 className="text-lg font-semibold text-text-primary">RMSE Leaderboard (Lower is Better)</h2>
          </div>
          <div className="flex-1 w-full relative">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={modelsData}
                layout="vertical"
                margin={{ top: 0, right: 30, left: 20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke={COLOURS.cardBorder} />
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={80} tick={{ fontSize: 12, fill: COLOURS.textSecondary, fontFamily: 'monospace' }} axisLine={false} tickLine={false} />
                <Tooltip
                  cursor={{ fill: 'transparent' }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-card-bg border border-card-border p-3 rounded-lg shadow-lg">
                          <p className="font-mono font-bold mb-1">{data.name}</p>
                          <p className="text-sm text-text-secondary">RMSE: <span className="text-brand-blue font-mono">{data.rmse?.toFixed(2)}</span></p>
                        </div>
                      )
                    }
                    return null;
                  }}
                />
                <Bar dataKey="rmse" radius={[0, 4, 4, 0]} barSize={32}>
                  {modelsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? COLOURS.riskNormal : COLOURS.brandBlue} fillOpacity={index === 0 ? 1 : 0.6} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        {/* Metrics Table */}
        <Card>
          <div className="flex items-center gap-2 mb-6">
            <Table size={20} className="text-brand-blue" />
            <h2 className="text-lg font-semibold text-text-primary">Detailed Metrics</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-card-border text-sm text-text-muted">
                  <th className="py-3 px-4 font-medium uppercase tracking-wider">Model</th>
                  <th className="py-3 px-4 font-medium uppercase tracking-wider text-right">RMSE</th>
                  <th className="py-3 px-4 font-medium uppercase tracking-wider text-right">MAE</th>
                  <th className="py-3 px-4 font-medium uppercase tracking-wider text-right">MAPE (%)</th>
                </tr>
              </thead>
              <tbody>
                {modelsData.map((model, idx) => (
                  <tr
                    key={model.name}
                    className={`border-b border-card-border last:border-0 ${idx === 0 ? 'bg-green-50/50 dark:bg-green-900/10' : 'hover:bg-surface/50'}`}
                  >
                    <td className="py-3 px-4 whitespace-nowrap">
                      <div className="flex items-center gap-2">
                        {idx === 0 && <Trophy size={14} className="text-risk-normal" />}
                        <span className={`font-mono text-sm ${idx === 0 ? 'font-bold text-risk-normal' : 'text-text-primary'}`}>{model.name}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 font-mono text-sm text-right text-text-secondary">{model.rmse?.toFixed(2) ?? '—'}</td>
                    <td className="py-3 px-4 font-mono text-sm text-right text-text-secondary">{model.mae?.toFixed(2) ?? '—'}</td>
                    <td className="py-3 px-4 font-mono text-sm text-right text-text-secondary">{model.mape?.toFixed(2) ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  )
}
