import React from 'react'
import clsx from 'clsx'

export default function ModelCorrelationHeatmap({ comparison }) {
  if (!comparison) {
    return (
      <div className="h-[250px] flex items-center justify-center border border-dashed border-card-border bg-surface/50 rounded-lg p-6 text-sm text-text-muted">
        No model comparison data available.
      </div>
    )
  }

  // If models agree/disagree stats are present, we can show a ranked list
  // or if there is a correlation matrix, we'll draw a heatmap
  const stats = comparison.model_agreement || comparison.stats || comparison
  
  // Try to find individual model scores or predictions format
  const modelStats = Object.entries(stats).filter(([key, val]) => 
    typeof val === 'object' && val !== null && !Array.isArray(val) && (val.anomaly_rate || val.detected_count || val.agreement)
  )

  if (modelStats.length === 0) {
    // If it's just raw scores, let's just show those
    const rawScores = Object.entries(stats).filter(([k,v]) => typeof v === 'number' && !k.includes('ensemble'))
    if (rawScores.length > 0) {
      return (
        <div className="flex flex-col h-full space-y-3 p-1">
          {rawScores.map(([model, score], i) => (
             <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-surface/40 border border-card-border">
               <span className="font-semibold text-text-primary uppercase text-sm">{model.replace('_', ' ')}</span>
               <span className={clsx("font-mono font-bold", score > 60 ? 'text-red-500' : 'text-emerald-500')}>
                 {score.toFixed(1)}
               </span>
             </div>
          ))}
          <p className="mt-auto pt-2 text-[11px] text-text-secondary leading-snug">
            <strong>Disagreement Check:</strong> Differences in scores show structural vs short-term variance.
          </p>
        </div>
      )
    }

    return (
      <div className="h-[250px] flex items-center justify-center border border-dashed border-card-border bg-surface/50 text-sm text-text-muted">
        Unrecognized comparison format.
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="space-y-3 flex-1 overflow-auto pr-2">
        {modelStats.map(([model, data], i) => {
           const rate = data.anomaly_rate || data.rate || 0
           const count = data.detected_count || data.count || '-'
           const pct = typeof rate === 'number' ? (rate * 100).toFixed(1) : '-'
           
           return (
             <div key={i} className="bg-surface/30 rounded-lg p-3 flex flex-col gap-2 border border-card-border hover:border-brand-blue/30 transition-colors">
               <div className="flex justify-between items-center">
                 <span className="font-semibold text-text-primary text-sm uppercase">{model.replace('_', ' ')}</span>
                 <span className="text-[10px] bg-white border border-card-border px-2 py-0.5 rounded font-mono text-text-secondary">Rate: {pct}%</span>
               </div>
               <div className="w-full bg-surface rounded-full h-1.5 overflow-hidden">
                 <div 
                   className={clsx("h-full rounded-full transition-all duration-1000", pct > 10 ? 'bg-amber-500' : 'bg-brand-blue')} 
                   style={{ width: `${Math.min(pct * 3, 100)}%` }}
                 />
               </div>
               <div className="text-[11px] text-text-muted flex justify-between">
                 <span>Anomalies flagged: {count}</span>
               </div>
             </div>
           )
        })}
      </div>
      
      <div className="mt-4 pt-3 border-t border-card-border/50">
        <p className="text-[11px] text-text-secondary leading-snug">
          High variance between models indicates complex regimes where traditional and ML methods disagree.
        </p>
      </div>
    </div>
  )
}
