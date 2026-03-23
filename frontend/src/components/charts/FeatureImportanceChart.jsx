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
          <CartesianGrid strokeDasharray="3 3" stroke="#2d2d2d" />
          <XAxis type="number" stroke="#888" />
          <YAxis type="category" dataKey="name" width={150} stroke="#888" fontSize={11} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #444' }}
            labelStyle={{ color: '#fff' }}
            formatter={(value) => value?.toFixed(4)}
          />
          <Bar dataKey="importance" fill="#3b82f6" radius={[0, 8, 8, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-text-secondary text-xs mt-4">
        Top features influencing price prediction. Higher values = stronger influence.
      </p>
    </Card>
  )
}
