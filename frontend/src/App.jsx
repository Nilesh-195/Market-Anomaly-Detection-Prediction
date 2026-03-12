import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { AlertTriangle, TrendingUp, Activity, Shield, RefreshCw, ChevronDown } from 'lucide-react'

const API = 'http://localhost:8000'

const ASSETS = ['SP500', 'VIX', 'BTC', 'GOLD', 'NASDAQ', 'TESLA']
const ASSET_LABELS = {
  SP500: 'S&P 500', VIX: 'VIX', BTC: 'Bitcoin',
  GOLD: 'Gold ETF', NASDAQ: 'NASDAQ 100', TESLA: 'Tesla',
}

const riskColor = (score) => {
  if (score >= 75) return '#ef4444'   // red — Extreme
  if (score >= 60) return '#f97316'   // orange — High Risk
  if (score >= 40) return '#eab308'   // yellow — Elevated
  return '#22c55e'                     // green — Normal
}

const riskBg = (score) => {
  if (score >= 75) return 'bg-red-900/30 border-red-700'
  if (score >= 60) return 'bg-orange-900/30 border-orange-700'
  if (score >= 40) return 'bg-yellow-900/30 border-yellow-700'
  return 'bg-green-900/30 border-green-700'
}

// ── Animated Gauge ────────────────────────────────────────────────────────────
function RiskGauge({ score }) {
  const angle = -135 + (score / 100) * 270
  const color = riskColor(score)
  return (
    <div className="flex flex-col items-center">
      <svg width="180" height="110" viewBox="0 0 180 110">
        <path d="M 20 100 A 70 70 0 0 1 160 100" fill="none" stroke="#1f2937" strokeWidth="16" strokeLinecap="round" />
        <path d="M 20 100 A 70 70 0 0 1 160 100" fill="none" stroke={color} strokeWidth="16"
              strokeLinecap="round" strokeDasharray={`${(score / 100) * 220} 220`} opacity="0.9" />
        <g transform={`translate(90,100) rotate(${angle})`}>
          <line x1="0" y1="0" x2="0" y2="-55" stroke="white" strokeWidth="3" strokeLinecap="round" />
          <circle cx="0" cy="0" r="5" fill="white" />
        </g>
        <text x="90" y="95" textAnchor="middle" fill="white" fontSize="22" fontWeight="bold">{Math.round(score)}</text>
      </svg>
      <span style={{ color }} className="font-semibold text-sm mt-1">
        {score >= 75 ? 'Extreme Anomaly' : score >= 60 ? 'High Risk' : score >= 40 ? 'Elevated' : 'Normal'}
      </span>
    </div>
  )
}

// ── Model Score Bar ───────────────────────────────────────────────────────────
function ModelBar({ label, score }) {
  const pct = Math.min(score, 100)
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span><span style={{ color: riskColor(pct) }}>{pct.toFixed(1)}</span>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-2">
        <div className="h-2 rounded-full transition-all duration-500"
             style={{ width: `${pct}%`, backgroundColor: riskColor(pct) }} />
      </div>
    </div>
  )
}

// ── Asset Card ────────────────────────────────────────────────────────────────
function AssetCard({ data, selected, onClick }) {
  if (!data) return <div className="bg-gray-800 rounded-xl p-4 animate-pulse h-24" />
  const { asset, ensemble_score: score, risk_label } = data
  return (
    <button onClick={onClick}
      className={`w-full text-left rounded-xl p-4 border transition-all duration-200 
        ${selected ? 'border-blue-500 bg-blue-900/20' : riskBg(score) + ' border'} hover:scale-[1.02]`}>
      <div className="flex justify-between items-start">
        <div>
          <div className="font-bold text-white">{ASSET_LABELS[asset]}</div>
          <div className="text-xs text-gray-400 mt-0.5">{asset}</div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold" style={{ color: riskColor(score) }}>{Math.round(score)}</div>
          <div className="text-xs" style={{ color: riskColor(score) }}>{risk_label}</div>
        </div>
      </div>
    </button>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [summary, setSummary]           = useState([])
  const [selected, setSelected]         = useState('SP500')
  const [analysis, setAnalysis]         = useState(null)
  const [forecast, setForecast]         = useState(null)
  const [history, setHistory]           = useState(null)
  const [comparison, setComparison]     = useState(null)
  const [loading, setLoading]           = useState(false)
  const [lastUpdated, setLastUpdated]   = useState(null)
  const [activeTab, setActiveTab]       = useState('overview')

  const fetchSummary = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/summary`)
      setSummary(r.data.assets)
    } catch (e) { console.error('Summary fetch failed', e) }
  }, [])

  const fetchAsset = useCallback(async (asset) => {
    setLoading(true)
    try {
      const [a, f, h, c] = await Promise.all([
        axios.get(`${API}/current-analysis/${asset}`),
        axios.get(`${API}/forecast/${asset}?days=10`),
        axios.get(`${API}/historical-anomalies/${asset}?top_n=20`),
        axios.get(`${API}/model-comparison/${asset}`),
      ])
      setAnalysis(a.data)
      setForecast(f.data)
      setHistory(h.data)
      setComparison(c.data)
      setLastUpdated(new Date().toLocaleTimeString())
    } catch (e) { console.error('Asset fetch failed', e) }
    setLoading(false)
  }, [])

  useEffect(() => { fetchSummary(); }, [fetchSummary])
  useEffect(() => { fetchAsset(selected); }, [selected, fetchAsset])

  // Build forecast chart data
  const forecastChartData = forecast?.forecast?.map(p => ({
    date: p.date.slice(5),   // MM-DD
    score: p.score,
    lower: p.lower,
    upper: p.upper,
  })) || []

  // Build history chart data (last 20 events)
  const historyChartData = (history?.events || []).slice(0, 15).reverse().map(e => ({
    date: e.date.slice(0, 7),
    score: e.ensemble_score,
  }))

  // Model comparison chart
  const modelChartData = comparison ? [
    { name: 'Z-Score',   score: comparison.stats.zscore_score?.mean || 0 },
    { name: 'Iso Forest',score: comparison.stats.iforest_score?.mean || 0 },
    { name: 'LSTM',      score: comparison.stats.lstm_score?.mean || 0 },
    { name: 'Prophet',   score: comparison.stats.prophet_score?.mean || 0 },
  ] : []

  return (
    <div className="min-h-screen bg-gray-950 text-white font-sans">
      {/* ── Header ── */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="text-blue-400" size={28} />
            <div>
              <h1 className="text-xl font-bold text-white">Market Anomaly Detection</h1>
              <p className="text-xs text-gray-500">Z-Score · Isolation Forest · LSTM · Prophet</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {lastUpdated && <span className="text-xs text-gray-500">Updated {lastUpdated}</span>}
            <button onClick={() => fetchAsset(selected)}
              className="flex items-center gap-1 text-xs bg-gray-800 hover:bg-gray-700 px-3 py-2 rounded-lg transition-colors">
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} /> Refresh
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

          {/* ── Left: Asset Selector ── */}
          <div className="lg:col-span-1">
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Assets</h2>
            <div className="space-y-2">
              {ASSETS.map(a => (
                <AssetCard key={a}
                  data={summary.find(s => s.asset === a)}
                  selected={selected === a}
                  onClick={() => setSelected(a)} />
              ))}
            </div>
          </div>

          {/* ── Right: Detail Panel ── */}
          <div className="lg:col-span-3 space-y-5">

            {/* Asset Header + Gauge */}
            <div className="bg-gray-900 rounded-2xl p-5 border border-gray-800">
              <div className="flex flex-col sm:flex-row items-center gap-6">
                <RiskGauge score={analysis?.ensemble_score || 0} />
                <div className="flex-1">
                  <h2 className="text-2xl font-bold">{ASSET_LABELS[selected]}</h2>
                  <p className="text-gray-400 text-sm mb-4">As of {analysis?.date || '—'}</p>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-800 rounded-lg p-3">
                      <div className="text-xs text-gray-500 mb-1">Ensemble Score</div>
                      <div className="text-2xl font-bold" style={{ color: riskColor(analysis?.ensemble_score || 0) }}>
                        {(analysis?.ensemble_score || 0).toFixed(1)}
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-3">
                      <div className="text-xs text-gray-500 mb-1">Risk Label</div>
                      <div className="text-lg font-bold" style={{ color: riskColor(analysis?.ensemble_score || 0) }}>
                        {analysis?.risk_label || '—'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Model breakdown bars */}
              {analysis && (
                <div className="mt-5 pt-4 border-t border-gray-800">
                  <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Model Scores</h3>
                  <div className="grid grid-cols-2 gap-x-6">
                    <ModelBar label="Z-Score"         score={analysis.model_scores.zscore} />
                    <ModelBar label="Isolation Forest" score={analysis.model_scores.iforest} />
                    <ModelBar label="LSTM Autoencoder" score={analysis.model_scores.lstm} />
                    <ModelBar label="Prophet Residual" score={analysis.model_scores.prophet} />
                  </div>
                </div>
              )}
            </div>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-gray-800 pb-0">
              {['overview', 'forecast', 'history', 'models'].map(tab => (
                <button key={tab} onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 text-sm font-medium rounded-t-lg capitalize transition-colors
                    ${activeTab === tab ? 'bg-gray-800 text-white border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-300'}`}>
                  {tab}
                </button>
              ))}
            </div>

            {/* ── Tab: Overview (All-Asset Scores) ── */}
            {activeTab === 'overview' && (
              <div className="bg-gray-900 rounded-2xl p-5 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
                  All Assets — Current Ensemble Score
                </h3>
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={summary.filter(s => s.ensemble_score !== undefined).map(s => ({
                    name: s.asset, score: s.ensemble_score
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                      labelStyle={{ color: '#f3f4f6' }}
                    />
                    <ReferenceLine y={60} stroke="#f97316" strokeDasharray="4 4" label={{ value: 'High Risk', fill: '#f97316', fontSize: 10 }} />
                    <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'Extreme', fill: '#ef4444', fontSize: 10 }} />
                    <Bar dataKey="score" radius={[4, 4, 0, 0]}
                      fill="#3b82f6"
                      label={{ position: 'top', fill: '#9ca3af', fontSize: 11 }}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* ── Tab: Forecast ── */}
            {activeTab === 'forecast' && (
              <div className="bg-gray-900 rounded-2xl p-5 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
                  10-Day Anomaly Score Forecast — {ASSET_LABELS[selected]}
                </h3>
                <ResponsiveContainer width="100%" height={240}>
                  <AreaChart data={forecastChartData}>
                    <defs>
                      <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                    <ReferenceLine y={60} stroke="#f97316" strokeDasharray="4 4" />
                    <ReferenceLine y={75} stroke="#ef4444" strokeDasharray="4 4" />
                    <Area type="monotone" dataKey="upper" stroke="none" fill="#3b82f6" fillOpacity={0.1} />
                    <Area type="monotone" dataKey="lower" stroke="none" fill="#111827" fillOpacity={1} />
                    <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2.5} dot={{ fill: '#3b82f6', r: 4 }} />
                  </AreaChart>
                </ResponsiveContainer>
                <div className="mt-4 grid grid-cols-5 gap-2">
                  {forecast?.forecast?.map((p, i) => (
                    <div key={i} className={`rounded-lg p-2 text-center border ${riskBg(p.score)}`}>
                      <div className="text-xs text-gray-400">{p.date.slice(5)}</div>
                      <div className="text-lg font-bold" style={{ color: riskColor(p.score) }}>{p.score.toFixed(0)}</div>
                      <div className="text-xs" style={{ color: riskColor(p.score) }}>{p.risk_label.split(' ')[0]}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Tab: History ── */}
            {activeTab === 'history' && (
              <div className="bg-gray-900 rounded-2xl p-5 border border-gray-800">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
                    Historical Anomaly Events — {ASSET_LABELS[selected]}
                  </h3>
                  <span className="text-xs text-gray-500">
                    {history?.total_anomaly_days || 0} total anomaly days
                  </span>
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={historyChartData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" horizontal={false} />
                    <XAxis type="number" domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
                    <YAxis type="category" dataKey="date" width={60} tick={{ fill: '#9ca3af', fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                    <Bar dataKey="score" radius={[0, 4, 4, 0]} fill="#ef4444" />
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-4 max-h-52 overflow-y-auto space-y-2 pr-1">
                  {history?.events?.map((e, i) => (
                    <div key={i} className={`flex justify-between items-center rounded-lg px-3 py-2 border ${riskBg(e.ensemble_score)}`}>
                      <div className="flex items-center gap-2">
                        <AlertTriangle size={13} style={{ color: riskColor(e.ensemble_score) }} />
                        <span className="text-sm font-mono text-gray-300">{e.date}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs" style={{ color: riskColor(e.ensemble_score) }}>{e.risk_label}</span>
                        <span className="font-bold text-sm" style={{ color: riskColor(e.ensemble_score) }}>
                          {e.ensemble_score.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Tab: Models ── */}
            {activeTab === 'models' && comparison && (
              <div className="bg-gray-900 rounded-2xl p-5 border border-gray-800">
                <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
                  Model Performance — {ASSET_LABELS[selected]}
                </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={modelChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                    <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#9ca3af', fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                    <Bar dataKey="score" fill="#8b5cf6" radius={[4, 4, 0, 0]}
                      label={{ position: 'top', fill: '#9ca3af', fontSize: 11, formatter: v => v.toFixed(1) }} />
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  {[
                    ['zscore_score', 'Z-Score'],
                    ['iforest_score', 'Isolation Forest'],
                    ['lstm_score', 'LSTM Autoencoder'],
                    ['prophet_score', 'Prophet Residual'],
                  ].map(([col, label]) => {
                    const s = comparison.stats[col] || {}
                    const corr = comparison.correlation_with_ensemble[col] || 0
                    return (
                      <div key={col} className="bg-gray-800 rounded-lg p-3">
                        <div className="text-sm font-semibold text-white mb-2">{label}</div>
                        <div className="grid grid-cols-2 gap-1 text-xs text-gray-400">
                          <span>Mean: <b className="text-white">{s.mean?.toFixed(1)}</b></span>
                          <span>Max: <b className="text-white">{s.max?.toFixed(1)}</b></span>
                          <span>P95: <b className="text-white">{s.p95?.toFixed(1)}</b></span>
                          <span>Corr: <b style={{ color: riskColor(corr * 100) }}>{corr.toFixed(3)}</b></span>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

          </div>
        </div>
      </main>
    </div>
  )
}
