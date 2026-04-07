import { useState } from 'react'
import { motion } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import TrainTestChart from '../components/evaluation/TrainTestChart'
import { 
  BarChart3, AlertCircle, Download
} from 'lucide-react'

const MODEL_LABELS = {
  zscore_score: 'Z-Score',
  iforest_score: 'Isolation Forest',
  lstm_score: 'LSTM Autoencoder',
  prophet_score: 'Prophet Residual',
  xgb_score: 'XGBoost Classifier',
  hmm_score: 'HMM Regime',
  tcn_score: 'TCN Autoencoder',
  vae_score: 'VAE Autoencoder',
  at_score: 'Anomaly Transformer',
  ensemble_score: 'Baseline Ensemble',
  adv_ensemble: 'Advanced Ensemble',
}

const MODEL_NAME_TO_KEY = Object.entries(MODEL_LABELS).reduce((acc, [key, label]) => {
  acc[label] = key
  return acc
}, {})

const PREFERRED_MODEL_ORDER = [
  'adv_ensemble',
  'ensemble_score',
  'xgb_score',
  'hmm_score',
  'at_score',
  'vae_score',
  'tcn_score',
  'lstm_score',
  'iforest_score',
  'zscore_score',
  'prophet_score',
]

function toNumber(value, fallback = null) {
  const num = Number(value)
  return Number.isFinite(num) ? num : fallback
}

function toPct(value, digits = 1) {
  const num = toNumber(value)
  if (num === null) return 'N/A'
  return `${(num * 100).toFixed(digits)}%`
}

function toFixedText(value, digits = 3) {
  const num = toNumber(value)
  if (num === null) return 'N/A'
  return num.toFixed(digits)
}

function normalizeEvaluationPayload(payload) {
  if (!payload || typeof payload !== 'object') return {}

  if (payload.asset_metrics && typeof payload.asset_metrics === 'object') {
    return payload.asset_metrics
  }

  if (Array.isArray(payload.rows)) {
    const metrics = {}
    payload.rows.forEach((row) => {
      const asset = row?.asset
      const modelName = row?.model
      if (!asset || !modelName) return

      const modelKey = MODEL_NAME_TO_KEY[modelName]
        || String(modelName).toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')

      if (!metrics[asset]) metrics[asset] = {}
      metrics[asset][modelKey] = {
        threshold: toNumber(row?.threshold),
        f1: toNumber(row?.f1),
        precision: toNumber(row?.precision),
        recall: toNumber(row?.recall),
        roc_auc: toNumber(row?.roc_auc),
        hit_rate: toNumber(row?.hit_rate),
        crashes_detected: toNumber(row?.crashes_detected, 0),
        crashes_in_range: toNumber(row?.crashes_in_range, 0),
        avg_lead_days: toNumber(row?.avg_lead_days),
      }
    })
    return metrics
  }

  if (payload.assets && typeof payload.assets === 'object' && !Array.isArray(payload.assets)) {
    return payload.assets
  }

  return payload
}

function getAccuracyValue(data) {
  const explicit = toNumber(data?.accuracy)
  if (explicit !== null) return explicit
  // Current backend report does not always include accuracy;
  // use event hit-rate as a concrete detection-accuracy fallback.
  return toNumber(data?.hit_rate)
}

const containerVariants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
}
const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 300, damping: 24 } }
}

export default function Evaluation({ evaluation, selectedAsset, loading }) {
  const [exportFormat, setExportFormat] = useState('csv')

  const assetsMetrics = normalizeEvaluationPayload(evaluation)
  const assetModels = assetsMetrics?.[selectedAsset] || {}

  const modelRowsForAsset = Object.entries(assetModels)
    .map(([modelKey, data]) => ({
      asset: selectedAsset,
      modelKey,
      model: MODEL_LABELS[modelKey] || modelKey,
      accuracy: getAccuracyValue(data),
      precision: toNumber(data?.precision),
      recall: toNumber(data?.recall),
      f1: toNumber(data?.f1),
      auc: toNumber(data?.roc_auc ?? data?.auc),
      threshold: toNumber(data?.threshold),
      hitRate: toNumber(data?.hit_rate),
      crashesDetected: toNumber(data?.crashes_detected, 0),
      crashesInRange: toNumber(data?.crashes_in_range, 0),
      avgLeadDays: toNumber(data?.avg_lead_days),
    }))
    .sort((a, b) => {
      const ia = PREFERRED_MODEL_ORDER.indexOf(a.modelKey)
      const ib = PREFERRED_MODEL_ORDER.indexOf(b.modelKey)
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib)
    })

  const allAssetModelRows = Object.entries(assetsMetrics || {}).flatMap(([asset, models]) => {
    if (!models || typeof models !== 'object') return []
    return Object.entries(models).map(([modelKey, data]) => ({
      asset,
      modelKey,
      model: MODEL_LABELS[modelKey] || modelKey,
      accuracy: getAccuracyValue(data),
      precision: toNumber(data?.precision),
      recall: toNumber(data?.recall),
      f1: toNumber(data?.f1),
      auc: toNumber(data?.roc_auc ?? data?.auc),
      threshold: toNumber(data?.threshold),
      hitRate: toNumber(data?.hit_rate),
      crashesDetected: toNumber(data?.crashes_detected, 0),
      crashesInRange: toNumber(data?.crashes_in_range, 0),
      avgLeadDays: toNumber(data?.avg_lead_days),
    }))
  })

  const exportMetrics = () => {
    const data = modelRowsForAsset.map((row) => ({
      asset: row.asset,
      model: row.model,
      accuracy: row.accuracy,
      precision: row.precision,
      recall: row.recall,
      f1: row.f1,
      roc_auc: row.auc,
      threshold: row.threshold,
      hit_rate: row.hitRate,
      crashes_detected: row.crashesDetected,
      crashes_in_range: row.crashesInRange,
      avg_lead_days: row.avgLeadDays,
    }))
    
    if (exportFormat === 'json') {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${selectedAsset}_evaluation.json`
      link.click()
    } else {
      const headers = [
        'asset', 'model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        'threshold', 'hit_rate', 'crashes_detected', 'crashes_in_range', 'avg_lead_days'
      ]
      const csvRows = [
        headers.join(','),
        ...data.map((row) => headers.map((h) => row[h] ?? '').join(',')),
      ]
      const csv = csvRows.join('\n')
      const blob = new Blob([csv], { type: 'text/csv' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `${selectedAsset}_evaluation.csv`
      link.click()
    }
  }

  return (
    <motion.div 
      className="space-y-6 pb-12"
      variants={containerVariants}
      initial="hidden"
      animate="show"
    >
      {/* Header */}
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-text-primary flex items-center gap-3 mb-2">
            <BarChart3 className="text-brand-blue" size={32} />
            Evaluation Metrics
          </h1>
          <p className="text-text-secondary">
            Real per-model evaluation scores for each asset, including classification and event-detection metrics.
          </p>
        </div>
        <div className="flex gap-2">
          <select 
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value)}
            className="px-3 py-2 rounded-lg border border-card-border bg-card-bg text-sm text-text-primary"
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
          </select>
          <button 
            onClick={exportMetrics}
            disabled={loading}
            className="px-4 py-2 bg-brand-blue text-white rounded-lg font-medium text-sm hover:bg-brand-blue-dim transition-colors flex items-center gap-2 disabled:opacity-50"
          >
            <Download size={16} />
            Export
          </button>
        </div>
      </motion.div>

      {/* Train vs Test Split Visualization */}
      <motion.div variants={itemVariants}>
        <TrainTestChart selectedAsset={selectedAsset} loading={loading} />
      </motion.div>

      {/* No Data State */}
      {modelRowsForAsset.length === 0 && !loading && (
        <Card className="p-12 flex flex-col items-center justify-center">
          <AlertCircle className="text-text-muted mb-4" size={48} />
          <h3 className="font-bold text-text-primary mb-1">No Evaluation Data</h3>
          <p className="text-text-muted text-sm">Metrics have not been computed for {selectedAsset}</p>
        </Card>
      )}

      {/* Selected Asset Model Matrix */}
      {modelRowsForAsset.length > 0 && (
        <motion.div variants={itemVariants}>
          <Card className="p-6">
            <h2 className="font-bold text-text-primary mb-4">{selectedAsset}: Per-Model Metrics</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-card-border">
                  <tr>
                    <th className="text-left py-2 px-3 font-bold text-text-primary">Model</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Accuracy</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Precision</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Recall</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">F1</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">AUC-ROC</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Hit Rate</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Threshold</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-card-border">
                  {modelRowsForAsset.map((row, i) => (
                    <tr key={i} className="hover:bg-surface/50">
                      <td className="py-3 px-3 font-medium text-text-primary">{row.model}</td>
                      <td className="text-right py-3 px-3 text-text-secondary">{toPct(row.accuracy)}</td>
                      <td className="text-right py-3 px-3 text-text-secondary">{toPct(row.precision)}</td>
                      <td className="text-right py-3 px-3 text-text-secondary">{toPct(row.recall)}</td>
                      <td className="text-right py-3 px-3 text-text-secondary">{toPct(row.f1)}</td>
                      <td className="text-right py-3 px-3 text-text-secondary">{toPct(row.auc)}</td>
                      <td className="text-right py-3 px-3 text-text-secondary">{toPct(row.hitRate)}</td>
                      <td className="text-right py-3 px-3">
                        <Badge variant={row.threshold >= 60 ? 'warning' : 'success'}>
                          {toFixedText(row.threshold, 1)}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </motion.div>
      )}

      {/* All Assets Model Matrix */}
      {allAssetModelRows.length > 0 && (
        <motion.div variants={itemVariants}>
          <Card className="p-6">
            <h2 className="font-bold text-text-primary mb-4">All Stocks: Per-Model Evaluation Metrics</h2>
            <div className="overflow-x-auto max-h-[520px]">
              <table className="w-full text-sm">
                <thead className="border-b border-card-border sticky top-0 bg-card-bg z-10">
                  <tr>
                    <th className="text-left py-2 px-3 font-bold text-text-primary">Asset</th>
                    <th className="text-left py-2 px-3 font-bold text-text-primary">Model</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Accuracy</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Precision</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Recall</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">F1</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">AUC-ROC</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Hit Rate</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Threshold</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Detected</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">In Range</th>
                    <th className="text-right py-2 px-3 font-bold text-text-primary">Avg Lead Days</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-card-border">
                  {allAssetModelRows.map((row, i) => (
                    <tr key={`${row.asset}-${row.modelKey}-${i}`} className="hover:bg-surface/50">
                      <td className="py-2 px-3 font-mono font-medium text-text-primary">{row.asset}</td>
                      <td className="py-2 px-3 text-text-primary">{row.model}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toPct(row.accuracy)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toPct(row.precision)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toPct(row.recall)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toPct(row.f1)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toPct(row.auc)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toPct(row.hitRate)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toFixedText(row.threshold, 1)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toFixedText(row.crashesDetected, 0)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toFixedText(row.crashesInRange, 0)}</td>
                      <td className="text-right py-2 px-3 text-text-secondary">{toFixedText(row.avgLeadDays, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </motion.div>
      )}
    </motion.div>
  )
}
