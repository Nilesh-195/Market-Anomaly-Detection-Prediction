import React from 'react'
import { Card } from '../ui/Card'
import clsx from 'clsx'

export default function EvaluationSnapshot({ evaluation, selectedAsset }) {
  if (!evaluation) {
    return (
      <div className="h-full flex items-center justify-center border border-dashed border-card-border bg-surface/50 rounded-lg p-6 text-sm text-text-muted">
        Evaluation data not available.
      </div>
    )
  }

  // Try to find stats for the selected asset
  const assetEval = evaluation[selectedAsset]

  if (!assetEval) {
    return (
      <div className="h-full flex items-center justify-center border border-dashed border-card-border bg-surface/50 rounded-lg p-6 text-sm text-text-muted">
        No evaluation snapshot found for {selectedAsset}.
      </div>
    )
  }

  // Check if it's the new standard structure with "models" or an ensemble
  // We'll prioritize looking for an "Ensemble" or "Global" or take the first model
  const models = assetEval.models || assetEval
  const modelNames = Object.keys(models).filter(k => typeof models[k] === 'object')
  
  if (modelNames.length === 0) {
    return (
      <div className="h-full flex flex-col justify-center text-sm text-text-muted">
        Unrecognized evaluation format.
      </div>
    )
  }

  // Pick the best representation to show: Ensemble if available, else first
  const targetModelName = modelNames.find(m => m.toLowerCase().includes('ensemble')) || modelNames[0]
  const metrics = models[targetModelName]

  const formatMetric = (val) => {
    if (typeof val !== 'number') return 'N/A'
    return val.toFixed(3)
  }

  const metricColor = (val) => {
    if (typeof val !== 'number') return 'text-text-muted'
    if (val >= 0.8) return 'text-emerald-600'
    if (val >= 0.6) return 'text-amber-600'
    return 'text-red-600'
  }

  return (
    <div className="flex flex-col h-full space-y-4">
      <div className="flex justify-between items-center text-sm">
        <span className="font-semibold text-text-primary capitalize">{targetModelName.replace('_', ' ')} Performance</span>
        <span className="text-[10px] uppercase font-mono tracking-wider px-2 py-1 bg-surface rounded text-text-secondary">Testing Set</span>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {[
          { label: 'Precision', val: metrics.precision || metrics.Precision, desc: 'True positives / predicted' },
          { label: 'Recall', val: metrics.recall || metrics.Recall, desc: 'Captured out of actual' },
          { label: 'F1 Score', val: metrics.f1 || metrics.f1_score || metrics['F1-Score'] || metrics['F1'], desc: 'Balances precision/recall' },
          { label: 'AUC-ROC', val: metrics.auc || metrics.auc_roc || metrics.roc_auc || metrics['ROC AUC'], desc: 'Ranking quality' }
        ].map((m, i) => (
          <div key={i} className="bg-surface/60 rounded-md p-3 border border-card-border/50">
            <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wide mb-1">{m.label}</div>
            <div className={clsx("font-mono text-xl font-bold", metricColor(m.val))}>
              {formatMetric(m.val)}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-auto pt-2 border-t border-card-border/50">
        <p className="text-[11px] text-text-secondary leading-snug">
          <strong>Tip:</strong> AUC measures ranking quality; F1 balances precision and recall. High numbers indicate stronger detection capability.
        </p>
      </div>
    </div>
  )
}
