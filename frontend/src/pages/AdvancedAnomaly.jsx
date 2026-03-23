/**
 * AdvancedAnomaly.jsx — Phase 2 Addition
 * 7-Model Ensemble Anomaly Detection with Advanced Features
 *
 * Shows:
 * - All 7 models breakdown (4 baseline + 3 advanced)
 * - Advanced ensemble vs baseline comparison
 * - Individual model scores with confidence metrics
 * - Risk assessment
 */

import { useState, useEffect } from 'react'
import { AlertTriangle, TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { API_BASE } from '../constants/config'
import clsx from 'clsx'

const MODEL_CONFIG = {
  baseline: [
    { id: 'zscore', name: 'Z-Score', desc: 'Statistical deviation detector', icon: '📊', color: 'blue' },
    { id: 'iforest', name: 'Isolation Forest', desc: 'Unsupervised anomaly detection', icon: '🌲', color: 'green' },
    { id: 'lstm', name: 'LSTM', desc: 'Deep learning sequence model', icon: '🔗', color: 'purple' },
    { id: 'prophet', name: 'Prophet', desc: 'Time series decomposition', icon: '📈', color: 'orange' },
  ],
  advanced: [
    { id: 'xgb', name: 'XGBoost', desc: 'Supervised crash prediction', icon: '⚡', color: 'red' },
    { id: 'hmm', name: 'HMM Regime', desc: 'Market state detector', icon: '🎭', color: 'indigo' },
    { id: 'tcn', name: 'TCN', desc: 'Temporal convolutional network', icon: '🌊', color: 'cyan' },
  ],
}

const RISK_THRESHOLDS = {
  LOW: { max: 40, color: 'text-green-400', bg: 'bg-green-500/20', label: 'Low Risk' },
  MEDIUM: { max: 60, color: 'text-amber-400', bg: 'bg-amber-500/20', label: 'Medium Risk' },
  HIGH: { max: 75, color: 'text-orange-400', bg: 'bg-orange-500/20', label: 'High Risk' },
  CRITICAL: { max: 100, color: 'text-red-400', bg: 'bg-red-500/20', label: 'Critical' },
}

function getRiskLevel(score) {
  if (score < RISK_THRESHOLDS.LOW.max) return RISK_THRESHOLDS.LOW
  if (score < RISK_THRESHOLDS.MEDIUM.max) return RISK_THRESHOLDS.MEDIUM
  if (score < RISK_THRESHOLDS.HIGH.max) return RISK_THRESHOLDS.HIGH
  return RISK_THRESHOLDS.CRITICAL
}

function ModelCard({ model, score }) {
  return (
    <Card>
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-2xl mb-1">{model.icon}</div>
          <h3 className="font-semibold text-text-primary">{model.name}</h3>
          <p className="text-xs text-text-secondary mt-1">{model.desc}</p>
        </div>
        <div className={clsx(
          'text-right p-2 rounded-lg',
          getRiskLevel(score).bg
        )}>
          <div className={clsx('font-mono font-bold text-lg', getRiskLevel(score).color)}>
            {score.toFixed(1)}
          </div>
        </div>
      </div>

      {/* Score bar */}
      <div className="w-full h-2 bg-surface rounded-full overflow-hidden">
        <div
          style={{ width: `${Math.min(score, 100)}%` }}
          className={clsx(
            'h-full transition-all',
            score < 40 && 'bg-green-500',
            score >= 40 && score < 60 && 'bg-amber-500',
            score >= 60 && score < 75 && 'bg-orange-500',
            score >= 75 && 'bg-red-500'
          )}
        />
      </div>
    </Card>
  )
}

export default function AdvancedAnomaly({ selectedAsset, loading: parentLoading }) {
  const [advancedData, setAdvancedData] = useState(null)
  const [baselineData, setBaselineData] = useState(null)
  const [tierComparison, setTierComparison] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!selectedAsset) return

    setLoading(true)
    setError(null)

    Promise.all([
      fetch(`${API_BASE}/anomaly/advanced/${selectedAsset}`).then(r => r.json()),
      fetch(`${API_BASE}/anomaly/current/${selectedAsset}`).then(r => r.json()),
      fetch(`${API_BASE}/anomaly/compare-tiers/${selectedAsset}`).then(r => r.json()),
    ])
      .then(([adv, base, comp]) => {
        setAdvancedData(adv)
        setBaselineData(base)
        setTierComparison(comp)
        setError(null)
      })
      .catch(err => setError(err?.message || 'Failed to load data'))
      .finally(() => setLoading(false))
  }, [selectedAsset])

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Advanced Anomaly Detection</h1>
          <p className="text-text-secondary text-sm">7-model ensemble with regime detection</p>
        </div>
        <Card>
          <div className="text-center py-8 text-text-secondary">
            <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-amber-500" />
            <p>{error}</p>
            <p className="text-sm mt-2">Run Phase 2 training to enable advanced anomaly detection.</p>
          </div>
        </Card>
      </div>
    )
  }

  const advScore = advancedData?.advanced_ensemble ?? 0
  const baseScore = baselineData?.ensemble_score ?? 0
  const improvement = advScore - baseScore
  const regimeText = advancedData?.current_regime ?? 'Unknown'
  const riskLevel = getRiskLevel(advScore)

  const isLoading = loading || parentLoading

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Advanced Anomaly Detection</h1>
          <p className="text-text-secondary text-sm">7-model ensemble with regime detection for {selectedAsset}</p>
        </div>
        <div className={clsx(
          'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium',
          riskLevel.bg + ' ' + riskLevel.color
        )}>
          <AlertTriangle size={16} />
          <span>{riskLevel.label}</span>
        </div>
      </div>

      {/* Top-level Ensemble Scores */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <p className="text-text-secondary text-sm uppercase tracking-wide mb-2">Baseline Ensemble</p>
          <div className={clsx(
            'font-mono font-bold text-3xl',
            getRiskLevel(baseScore).color
          )}>
            {baseScore.toFixed(1)}
          </div>
          <p className="text-text-secondary text-xs mt-2">4 models (Z-Score, IForest, LSTM, Prophet)</p>
        </Card>

        <Card className="bg-blue-500/10 border-blue-600">
          <p className="text-text-secondary text-sm uppercase tracking-wide mb-2">Advanced Ensemble</p>
          <div className={clsx(
            'font-mono font-bold text-3xl',
            getRiskLevel(advScore).color
          )}>
            {advScore.toFixed(1)}
          </div>
          <p className="text-text-secondary text-xs mt-2">7 models (baseline + XGB, HMM, TCN)</p>
        </Card>

        <Card>
          <p className="text-text-secondary text-sm uppercase tracking-wide mb-2">Improvement</p>
          <div className={clsx(
            'font-mono font-bold text-3xl',
            improvement > 0 ? 'text-orange-400' : improvement < 0 ? 'text-green-400' : 'text-text-secondary'
          )}>
            {improvement > 0 ? '+' : ''}{improvement.toFixed(1)}
          </div>
          <p className="text-text-secondary text-xs mt-2">Advanced vs Baseline delta</p>
        </Card>
      </div>

      {/* Market Regime */}
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-4">Current Market Regime</h3>
            <div className={clsx(
              'inline-flex items-center gap-2 px-4 py-3 rounded-lg text-base font-medium',
              regimeText === 'bull' && 'bg-green-500/20 text-green-400',
              regimeText === 'bear' && 'bg-amber-500/20 text-amber-400',
              regimeText === 'crisis' && 'bg-red-500/20 text-red-400'
            )}>
              {regimeText === 'bull' && <TrendingUp size={20} />}
              {regimeText === 'bear' && <TrendingDown size={20} />}
              {regimeText === 'crisis' && <AlertTriangle size={20} />}
              <span className="uppercase">{regimeText} REGIME</span>
            </div>
            <p className="text-text-secondary text-sm mt-4">
              The market is currently in a <strong>{regimeText}</strong> regime based on HMM analysis.
              This regime detection helps contextualize anomaly scores.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold text-text-primary mb-4">What This Means</h3>
            <div className="space-y-2 text-sm text-text-secondary">
              {regimeText === 'bull' && (
                <>
                  <p>✓ Low volatility environment</p>
                  <p>✓ Returns typically positive</p>
                  <p>⚠ Anomalies may indicate reversals</p>
                </>
              )}
              {regimeText === 'bear' && (
                <>
                  <p>⚠ Moderate volatility</p>
                  <p>⚠ Risk of further downside</p>
                  <p>✓ Caution warranted</p>
                </>
              )}
              {regimeText === 'crisis' && (
                <>
                  <p className="text-red-400">⚠ High volatility environment</p>
                  <p className="text-red-400">⚠ Significant risk present</p>
                  <p className="text-red-400">⚠ Immediate attention required</p>
                </>
              )}
            </div>
          </div>
        </div>
      </Card>

      {/* Baseline Models (4) */}
      <div>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Baseline Models (4)</h2>
          <p className="text-text-secondary text-sm">Classical and LSTM approaches</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {MODEL_CONFIG.baseline.map(model => {
            const score = advancedData?.model_scores?.[model.id] ?? 0
            return <ModelCard key={model.id} model={model} score={score} />
          })}
        </div>
      </div>

      {/* Advanced Models (3) */}
      <div>
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-text-primary">Advanced Models (3) — Phase 2</h2>
          <p className="text-text-secondary text-sm">Supervised, regime-aware, and deep temporal approaches</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {MODEL_CONFIG.advanced.map(model => {
            const score = advancedData?.model_scores?.[model.id] ?? 0
            return <ModelCard key={model.id} model={model} score={score} />
          })}
        </div>
      </div>

      {/* Model Descriptions */}
      <Card>
        <h3 className="text-lg font-semibold text-text-primary mb-4">Model Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-text-primary mb-3">Baseline Tier</h4>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li><strong>Z-Score:</strong> Detects price deviation from moving average</li>
              <li><strong>Isolation Forest:</strong> Unsupervised anomaly in high-dimensional feature space</li>
              <li><strong>LSTM:</strong> Learns temporal patterns and identifies breaks</li>
              <li><strong>Prophet:</strong> Decomposes trend/seasonality and finds residual anomalies</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-text-primary mb-3">Advanced Tier (Phase 2)</h4>
            <ul className="space-y-2 text-sm text-text-secondary">
              <li><strong>XGBoost:</strong> Supervised classifier trained on historical crash labels</li>
              <li><strong>HMM Regime:</strong> Detects bull/bear/crisis market states</li>
              <li><strong>TCN:</strong> Temporal Convolutional Network for deep sequence modeling</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Ensemble Strategy */}
      <Card>
        <h3 className="text-lg font-semibold text-text-primary mb-4">Ensemble Strategy</h3>
        <div className="space-y-4 text-sm text-text-secondary">
          <div>
            <p className="font-medium text-text-primary mb-1">Dynamic Weighting</p>
            <p>Each model contributes to the final score with learned weights optimized for crash detection:</p>
            <ul className="list-disc list-inside mt-2 space-y-1 text-xs">
              <li>Z-Score: 5% (baseline statistical)</li>
              <li>Isolation Forest: 10% (unsupervised ML)</li>
              <li>LSTM: 20% (deep sequence learning)</li>
              <li>Prophet: 10% (trend deviation)</li>
              <li>XGBoost: 30% (supervised — highest weight)</li>
              <li>HMM: 10% (market regime context)</li>
              <li>TCN: 15% (temporal conv network)</li>
            </ul>
          </div>
          <div className="p-3 bg-surface rounded-lg">
            <p>The advanced ensemble prioritizes <strong>supervised signals (XGBoost)</strong> which has seen historical crash events,
            combined with <strong>temporal deep learning (LSTM, TCN)</strong> for pattern detection and
            <strong>regime awareness (HMM)</strong> for context.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}
