import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Activity, ShieldAlert, TrendingUp, BarChart2, Network, ArrowRight } from 'lucide-react'

export default function HeroLanding({ onLaunch }) {
  const [ticks, setTicks] = useState([])

  useEffect(() => {
    // Simulate real-time signal chart
    const interval = setInterval(() => {
      setTicks(prev => {
        const next = [...prev, Math.random() * 100]
        if (next.length > 50) next.shift()
        return next
      })
    }, 150)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-page-bg text-text-primary flex flex-col relative overflow-hidden transition-colors duration-500">
      {/* Background accents */}
      <div className="absolute inset-0 pointer-events-none" style={{ backgroundImage: 'radial-gradient(var(--card-border) 1px, transparent 1px)', backgroundSize: '40px 40px', opacity: 0.3 }} />
      <div className="absolute -top-[20%] -right-[10%] w-[600px] h-[600px] bg-brand-blue/20 blur-[120px] rounded-full pointer-events-none" />
      <div className="absolute -bottom-[20%] -left-[10%] w-[600px] h-[600px] bg-chart-purple/20 blur-[120px] rounded-full pointer-events-none" />

      {/* Nav */}
      <nav className="flex items-center justify-between p-6 relative z-10 max-w-7xl w-full mx-auto">
        <div className="flex items-center gap-3">
          <Activity size={24} className="text-brand-blue" />
          <span className="font-mono font-bold tracking-wider text-xl">MARKET ANOMALY</span>
        </div>
      </nav>

      {/* Hero Content */}
      <main className="flex-1 flex flex-col items-center justify-center relative z-10 px-6 max-w-5xl mx-auto w-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-6"
        >
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-brand-blue/30 bg-brand-blue/10 text-brand-blue text-sm font-medium mb-4">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-blue opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-blue"></span>
            </span>
            Real-time Detection Engine V2.0
          </div>
          
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight">
            Anticipate the Market. <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-blue to-chart-purple">Detect the Unseen.</span>
          </h1>
          
          <p className="text-xl text-text-secondary max-w-2xl mx-auto">
            Advanced ensemble model combining Deep Learning and Statistical Analysis to forecast price movements and identify critical structural anomalies before they happen.
          </p>

          {/* Model Badges */}
          <div className="flex flex-wrap items-center justify-center gap-3 pt-4">
            {['LSTM', 'Prophet', 'Isolation Forest', 'VAR', 'Z-Score'].map(model => (
              <span key={model} className="px-4 py-2 rounded-xl bg-surface border border-card-border text-sm font-mono font-medium shadow-sm">
                {model}
              </span>
            ))}
          </div>

          <div className="pt-10">
            <button
              onClick={onLaunch}
              className="group flex items-center gap-3 bg-brand-blue hover:bg-brand-blue-dim text-white px-8 py-4 rounded-xl font-bold text-lg transition-all hover:scale-105 active:scale-95 shadow-lg shadow-brand-blue/25"
            >
              Launch Platform
              <ArrowRight className="group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </motion.div>

        {/* Live Signal Chart Simulation */}
        <div className="mt-20 w-full max-w-3xl h-32 flex items-end gap-1 opacity-50 relative">
          <div className="absolute inset-0 bg-gradient-to-t from-page-bg via-transparent to-transparent z-10 pointer-events-none" />
          {ticks.map((val, i) => (
            <div
              key={i}
              className="flex-1 bg-brand-blue rounded-t-sm transition-all duration-150"
              style={{ height: `${val}%` }}
            />
          ))}
        </div>
      </main>
    </div>
  )
}
