import { useMemo } from 'react'
import { Card } from '../ui/Card'

/**
 * AttentionHeatmap - Visualizes LSTM Seq2Seq attention weights
 * Shows which past days influenced each future prediction
 * X-axis: Past days (lookback window)
 * Y-axis: Future prediction days (forecast horizon)
 * Color intensity: Attention weight (influence)
 */
export default function AttentionHeatmap({ weights, title = 'LSTM Attention Weights' }) {
  const heatmapData = useMemo(() => {
    if (!weights || weights.length === 0) return null

    // weights is typically (horizon, lookback)
    const height = weights.length
    const width = weights[0]?.length || 30

    return { weights: Array.isArray(weights) ? weights : [], height, width }
  }, [weights])

  if (!heatmapData || heatmapData.weights.length === 0) {
    return (
      <Card>
        <p className="text-text-muted text-sm">No attention weights available</p>
      </Card>
    )
  }

  // Normalize weights for color intensity (0-1)
  let maxWeight = 0
  heatmapData.weights.forEach(row => {
    if (Array.isArray(row)) {
      const rowMax = Math.max(...row.map(w => Math.abs(w)))
      maxWeight = Math.max(maxWeight, rowMax)
    }
  })

  const getColor = (weight) => {
    if (maxWeight === 0) return '#1a1a1a'
    const normalized = Math.abs(weight) / maxWeight
    // Blue gradient: light to dark based on intensity
    const intensity = Math.round(normalized * 255)
    return `rgba(59, 130, 246, ${normalized})`  // Blue with alpha based on intensity
  }

  const cellSize = Math.min(20, 100 / heatmapData.width)
  const rowHeight = Math.min(20, 300 / heatmapData.height)

  return (
    <Card>
      <h3 className="text-lg font-semibold text-text-primary mb-4">{title}</h3>

      <div className="overflow-x-auto">
        <div className="inline-block">
          {/* Heatmap */}
          <div className="flex flex-col items-center">
            {/* X-axis labels */}
            <div className="flex mb-1">
              <div style={{ width: '60px' }} />
              <div className="flex gap-0">
                {heatmapData.weights[0]?.map((_, j) => (
                  <div
                    key={j}
                    style={{ width: `${cellSize}px`, textAlign: 'center', fontSize: '10px' }}
                    className="text-text-secondary"
                  >
                    {j % 5 === 0 ? j - heatmapData.width : ''}
                  </div>
                ))}
              </div>
            </div>

            {/* Grid */}
            {heatmapData.weights.map((row, i) => (
              <div key={i} className="flex items-center">
                {/* Y-axis label */}
                <div
                  style={{ width: '60px', height: `${rowHeight}px` }}
                  className="text-text-secondary text-xs flex items-center justify-end pr-2"
                >
                  +{i + 1}d
                </div>

                {/* Cells */}
                <div className="flex gap-0">
                  {Array.isArray(row) ? row.map((weight, j) => (
                    <div
                      key={`${i}-${j}`}
                      title={`Day {-${heatmapData.width - j}} → +${i + 1}d: ${weight?.toFixed(3)}`}
                      style={{
                        width: `${cellSize}px`,
                        height: `${rowHeight}px`,
                        backgroundColor: getColor(weight),
                        border: '1px solid #333',
                      }}
                    />
                  )) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-card-border">
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>Weak</span>
          <div className="flex gap-1">
            {[0.2, 0.4, 0.6, 0.8, 1.0].map(v => (
              <div
                key={v}
                style={{
                  width: '16px',
                  height: '16px',
                  backgroundColor: `rgba(59, 130, 246, ${v})`,
                  border: '1px solid #333',
                }}
              />
            ))}
          </div>
          <span>Strong</span>
        </div>
      </div>

      <p className="text-text-secondary text-xs mt-4">
        <strong>X-axis:</strong> Past days (relative to forecast start)
        <br />
        <strong>Y-axis:</strong> Future prediction days
        <br />
        <strong>Color intensity:</strong> How much each past day influenced that day's prediction
      </p>
    </Card>
  )
}
