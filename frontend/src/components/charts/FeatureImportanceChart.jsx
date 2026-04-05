import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Card } from '../ui/Card'

/**
 * FeatureImportanceChart - Visualizes SHAP values or feature importance from XGBoost
 * Shows top features that drive price predictions
 */
export default function FeatureImportanceChart({ importance, title = 'Feature Importance' }) {
  const data = useMemo(() => {
    if (!importance || Object.keys(importance).length === 0) return []

    return Object.entries(importance)
      .slice(0, 10)  // Top 10 features
      .map(([name, value]) => ({
        name: name.replace(/_/g, ' ').toUpperCase(),
        importance: typeof value === 'number' ? value : 0,
      }))
      .sort((a, b) => b.importance - a.importance)
  }, [importance])

  if (data.length === 0) {
    return (
      <Card>
        <p className="text-text-muted text-sm">No feature importance data available</p>
      </Card>
    )
  }

  return (
    <Card>
      <h3 className="text-lg font-semibold text-text-primary mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="4 4" stroke="#E2E8F0" horizontal={true} vertical={false} />
          <XAxis type="number" stroke="#94A3B8" fontSize={11} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={100} stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: '#F1F5F9' }}
            contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)', border: '1px solid #E2E8F0', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.05)', color: '#0F172A' }}
            labelStyle={{ color: '#475569', fontSize: '12px', fontWeight: 'bold' }}
            formatter={(value) => [`${value?.toFixed(2)}%`, 'Importance']}
            animationDuration={300}
          />
          <Bar dataKey="importance" fill="url(#blueGrad)" radius={[0, 6, 6, 0]} animationDuration={1000} />
          <defs>
            <linearGradient id="blueGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.8} />
              <stop offset="100%" stopColor="#0EA5E9" stopOpacity={1} />
            </linearGradient>
          </defs>
        </BarChart>
      </ResponsiveContainer>
      <p className="text-text-secondary text-xs mt-4">
        Top features influencing price prediction. Higher values = stronger influence.
      </p>
    </Card>
  )
}
