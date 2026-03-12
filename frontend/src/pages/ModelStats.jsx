import { useState } from 'react'
import { motion } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import ROCCurves from '../components/charts/ROCCurves'
import { formatScore } from '../utils/formatters'
import { COLOURS } from '../constants/colours'
import clsx from 'clsx'

// evaluation shape: { SP500: { zscore_score: {precision,recall,f1,roc_auc,hit_rate,...}, ... }, ... }

const TABS       = ['All Models', 'Z-Score', 'Isolation Forest', 'LSTM', 'Prophet']
const MODEL_KEYS = ['zscore_score', 'iforest_score', 'lstm_score', 'prophet_score']
const MODEL_META = {
  zscore_score:  { label: 'Z-Score',           variant: 'blue'   },
  iforest_score: { label: 'Isolation Forest',  variant: 'purple' },
  lstm_score:    { label: 'LSTM Autoencoder',  variant: 'cyan'   },
  prophet_score: { label: 'Prophet',           variant: 'green'  },
}

// Average a metric across all assets for a given model key
function avgForModel(evaluation, modelKey, metric) {
  if (!evaluation) return null
  const vals = Object.values(evaluation)
    .map(asset => asset?.[modelKey]?.[metric])
    .filter(v => v != null)
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
}

// Average across all models and all assets
function avgAll(evaluation, metric) {
  if (!evaluation) return null
  const vals = []
  for (const asset of Object.values(evaluation)) {
    for (const model of Object.values(asset)) {
      if (model?.[metric] != null) vals.push(model[metric])
    }
  }
  return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
}

function fmt(v, decimals = 3) {
  return v == null ? '—' : v.toFixed(decimals)
}

function FeatureBar({ name, importance, color }) {
  return (
    <div className="flex items-center gap-3">
      <span className="text-[#64748B] text-xs font-mono w-32 truncate">{name}</span>
      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${importance * 100}%` }}
          transition={{ duration: 0.6 }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
      <span className="text-[#64748B] text-xs font-mono w-10 text-right">
        {(importance * 100).toFixed(1)}%
      </span>
    </div>
  )
}

const FEATURES = [
  { name: 'zscore_20',     importance: 0.18, color: COLOURS.chartBlue   },
  { name: 'vol_30',        importance: 0.15, color: COLOURS.chartGreen  },
  { name: 'drawdown',      importance: 0.13, color: COLOURS.chartGreen  },
  { name: 'log_return',    importance: 0.11, color: COLOURS.chartBlue   },
  { name: 'rsi_14',        importance: 0.09, color: COLOURS.chartPurple },
  { name: 'bb_position',   importance: 0.08, color: COLOURS.chartPurple },
  { name: 'vol_ratio',     importance: 0.08, color: COLOURS.chartGreen  },
  { name: 'volume_zscore', importance: 0.07, color: COLOURS.riskHigh    },
  { name: 'macd_hist',     importance: 0.06, color: COLOURS.chartPurple },
  { name: 'atr_ratio',     importance: 0.05, color: COLOURS.chartGreen  },
]

export default function ModelStats({ evaluation, loading }) {
  const [activeTab, setActiveTab] = useState('All Models')
  const assets = evaluation ? Object.keys(evaluation) : []

  // Active model key based on tab
  const tabModelKey = {
    'Z-Score':          'zscore_score',
    'Isolation Forest': 'iforest_score',
    'LSTM':             'lstm_score',
    'Prophet':          'prophet_score',
  }[activeTab] ?? null  // null = All Models

  // KPI values: average across all assets for selected model (or all models)
  const kpiMetrics = tabModelKey
    ? {
        precision: avgForModel(evaluation, tabModelKey, 'precision'),
        recall:    avgForModel(evaluation, tabModelKey, 'recall'),
        f1:        avgForModel(evaluation, tabModelKey, 'f1'),
        auc:       avgForModel(evaluation, tabModelKey, 'roc_auc'),
      }
    : {
        precision: avgAll(evaluation, 'precision'),
        recall:    avgAll(evaluation, 'recall'),
        f1:        avgAll(evaluation, 'f1'),
        auc:       avgAll(evaluation, 'roc_auc'),
      }

  return (
    <div className="space-y-4">
      {/* Header + model tabs */}
      <div>
        <h1 className="text-[#F1F5F9] text-xl font-semibold">Model Statistics</h1>
        <p className="text-[#64748B] text-sm mt-0.5">Performance metrics across all assets</p>
      </div>

      <div className="flex gap-1 bg-surface border border-card-border rounded-xl p-1 w-fit">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
              activeTab === t
                ? 'bg-brand-blue text-white'
                : 'text-[#64748B] hover:text-[#F1F5F9]'
            )}
          >
            {t}
          </button>
        ))}
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Avg Precision', key: 'precision', color: COLOURS.chartBlue   },
          { label: 'Avg Recall',    key: 'recall',    color: COLOURS.chartPurple },
          { label: 'Avg F1 Score',  key: 'f1',        color: COLOURS.chartCyan   },
          { label: 'Avg AUC (ROC)', key: 'auc',       color: COLOURS.riskNormal  },
        ].map((m, i) => (
          <motion.div
            key={m.key}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06 }}
          >
            <Card hover>
              <div className="text-[#64748B] text-[11px] uppercase tracking-wider mb-2">{m.label}</div>
              <div className="font-mono font-bold text-2xl" style={{ color: m.color }}>
                {loading ? '—' : fmt(kpiMetrics[m.key])}
              </div>
              <div className="text-[#334155] text-xs mt-1">
                {tabModelKey ? MODEL_META[tabModelKey]?.label : `${assets.length} assets`}
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* ROC Curves */}
      <Card>
        <div className="text-[#F1F5F9] font-medium mb-1">ROC Curves — All Assets</div>
        <div className="text-[#64748B] text-xs mb-4">Higher AUC = better anomaly discrimination</div>
        {loading
          ? <div className="h-[300px] bg-surface rounded-lg animate-pulse" />
          : <ROCCurves evaluation={evaluation} />
        }
      </Card>

      {/* Per-asset early warning table + feature importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <div className="text-[#F1F5F9] font-medium mb-1">Early Warning by Asset & Model</div>
          <div className="text-[#64748B] text-xs mb-4">Hit rate = crashes detected before event</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-card-border">
                {['Asset', 'Model', 'AUC', 'Hit Rate', 'Detected'].map(h => (
                  <th key={h} className="text-left text-[#64748B] text-[11px] uppercase tracking-wider pb-2 px-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 8 }).map((_, i) => (
                    <tr key={i} className="border-b border-card-border/50">
                      {Array.from({ length: 5 }).map((_, j) => (
                        <td key={j} className="py-2 px-2">
                          <div className="h-3 w-12 bg-surface rounded animate-pulse" />
                        </td>
                      ))}
                    </tr>
                  ))
                : assets.flatMap(asset =>
                    MODEL_KEYS.map(mk => {
                      const d        = evaluation?.[asset]?.[mk]
                      if (!d) return null
                      const hit      = d.hit_rate ?? 0
                      const auc      = d.roc_auc  ?? 0
                      const detected = d.crashes_detected ?? 0
                      const total    = d.crashes_in_range ?? 0
                      const hitVariant = hit > 0.5 ? 'green' : hit > 0.25 ? 'yellow' : 'red'
                      return (
                        <tr key={`${asset}-${mk}`} className="border-b border-card-border/50 hover:bg-surface transition-colors">
                          <td className="py-1.5 px-2 font-mono text-xs text-[#F1F5F9]">{asset}</td>
                          <td className="py-1.5 px-2">
                            <Badge variant={MODEL_META[mk].variant}>{MODEL_META[mk].label}</Badge>
                          </td>
                          <td className="py-1.5 px-2 font-mono text-xs"
                              style={{ color: auc > 0.75 ? COLOURS.riskNormal : COLOURS.riskElevated }}>
                            {fmt(auc)}
                          </td>
                          <td className="py-1.5 px-2">
                            <Badge variant={hitVariant}>{(hit * 100).toFixed(0)}%</Badge>
                          </td>
                          <td className="py-1.5 px-2 text-[#64748B] text-xs font-mono">
                            {detected}/{total}
                          </td>
                        </tr>
                      )
                    }).filter(Boolean)
                  )
              }
            </tbody>
          </table>
        </Card>

        <Card>
          <div className="text-[#F1F5F9] font-medium mb-1">Feature Importance</div>
          <div className="text-[#64748B] text-xs mb-4">Isolation Forest — relative contribution</div>
          <div className="space-y-2.5">
            {FEATURES.map(f => (
              <FeatureBar key={f.name} {...f} />
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}
