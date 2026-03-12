import { useState } from 'react'
import { motion } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import ROCCurves from '../components/charts/ROCCurves'
import { formatScore } from '../utils/formatters'
import { COLOURS } from '../constants/colours'
import clsx from 'clsx'

const TABS = ['Ensemble', 'Z-Score', 'Isolation Forest', 'LSTM', 'Prophet']
const CRASH_EVENTS = [
  { name: 'Flash Crash',          date: '2010-05-06' },
  { name: 'US Debt Downgrade',    date: '2011-08-08' },
  { name: 'China Black Monday',   date: '2015-08-24' },
  { name: 'Volmageddon',          date: '2018-02-05' },
  { name: 'Christmas Crash',      date: '2018-12-24' },
  { name: 'COVID First Wave',     date: '2020-02-24' },
  { name: 'COVID Peak Crash',     date: '2020-03-16' },
  { name: 'GameStop Squeeze',     date: '2021-01-28' },
  { name: 'Fed Tightening Panic', date: '2022-01-24' },
  { name: 'Luna/Terra Collapse',  date: '2022-05-12' },
  { name: 'UK Gilt Crisis',       date: '2022-09-28' },
  { name: 'SVB Bank Collapse',    date: '2023-03-10' },
  { name: 'Yen Carry Unwind',     date: '2024-08-05' },
]

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
  { name: 'zscore_20',      importance: 0.18, color: COLOURS.chartBlue   },
  { name: 'vol_30',         importance: 0.15, color: COLOURS.chartGreen  },
  { name: 'drawdown',       importance: 0.13, color: COLOURS.chartGreen  },
  { name: 'log_return',     importance: 0.11, color: COLOURS.chartBlue   },
  { name: 'rsi_14',         importance: 0.09, color: COLOURS.chartPurple },
  { name: 'bb_position',    importance: 0.08, color: COLOURS.chartPurple },
  { name: 'vol_ratio',      importance: 0.08, color: COLOURS.chartGreen  },
  { name: 'volume_zscore',  importance: 0.07, color: COLOURS.riskHigh    },
  { name: 'macd_hist',      importance: 0.06, color: COLOURS.chartPurple },
  { name: 'atr_ratio',      importance: 0.05, color: COLOURS.chartGreen  },
]

export default function ModelStats({ evaluation, loading }) {
  const [activeTab, setActiveTab] = useState('Ensemble')

  const assets = evaluation ? Object.keys(evaluation) : []
  const getMetric = (metric) => {
    if (!evaluation) return '—'
    const vals = assets.map(a => evaluation[a]?.[metric]).filter(v => v != null)
    if (!vals.length) return '—'
    return formatScore(vals.reduce((s, v) => s + v, 0) / vals.length)
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
          { label: 'Avg AUC',       key: 'auc_score', color: COLOURS.riskNormal  },
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
                {loading ? '—' : getMetric(m.key)}
              </div>
              <div className="text-[#334155] text-xs mt-1">across {assets.length} assets</div>
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

      {/* Per-asset table + feature importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <div className="text-[#F1F5F9] font-medium mb-1">Early Warning Analysis</div>
          <div className="text-[#64748B] text-xs mb-4">Crash detection results by asset</div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-card-border">
                {['Asset', 'AUC', 'Hit Rate', 'Detected'].map(h => (
                  <th key={h} className="text-left text-[#64748B] text-[11px] uppercase tracking-wider pb-2 px-2">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {loading
                ? Array.from({ length: 6 }).map((_, i) => (
                    <tr key={i} className="border-b border-card-border/50">
                      {Array.from({ length: 4 }).map((_, j) => (
                        <td key={j} className="py-2.5 px-2">
                          <div className="h-3 w-16 bg-surface rounded animate-pulse" />
                        </td>
                      ))}
                    </tr>
                  ))
                : assets.map(asset => {
                    const d = evaluation[asset]
                    const hit = d?.hit_rate ?? 0
                    const auc = d?.auc_score ?? 0
                    const hitVariant = hit > 0.5 ? 'green' : hit > 0.25 ? 'yellow' : 'red'
                    return (
                      <tr key={asset} className="border-b border-card-border/50 hover:bg-surface transition-colors">
                        <td className="py-2.5 px-2 font-mono font-medium text-xs text-[#F1F5F9]">{asset}</td>
                        <td className="py-2.5 px-2 font-mono text-xs" style={{ color: auc > 0.75 ? COLOURS.riskNormal : COLOURS.riskElevated }}>
                          {auc.toFixed(3)}
                        </td>
                        <td className="py-2.5 px-2">
                          <Badge variant={hitVariant}>{(hit * 100).toFixed(0)}%</Badge>
                        </td>
                        <td className="py-2.5 px-2 text-[#64748B] text-xs font-mono">
                          {d?.detected ?? '—'}/{d?.total ?? '—'}
                        </td>
                      </tr>
                    )
                  })
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
