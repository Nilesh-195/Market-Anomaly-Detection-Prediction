import React, { useEffect } from 'react'
import MinimalNav from '../components/landing/MinimalNav'
import Hero from '../components/landing/Hero'
import FeatureGrid from '../components/landing/FeatureGrid'
import HowItWorks from '../components/landing/HowItWorks'
import FinalCTA from '../components/landing/FinalCTA'
import { motion, useScroll, useTransform } from 'framer-motion'

export default function Landing() {
  const { scrollYProgress } = useScroll()
  const opacity = useTransform(scrollYProgress, [0, 0.05], [1, 0])

  // Scroll to top on mount
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  return (
    <div className="min-h-screen bg-page-bg font-sans selection:bg-brand-blue/20">
      
      {/* Navigation */}
      <MinimalNav />

      {/* Main Content Sections */}
      <main className="w-full flex flex-col items-center overflow-x-hidden">
        <Hero />
        <FeatureGrid />
        
        {/* Showcase Panel */}
        <section id="models" className="py-24 px-6 max-w-7xl mx-auto w-full">
          <div className="bg-card-bg border border-card-border rounded-3xl p-8 mb-12 shadow-sm flex flex-col md:flex-row items-center gap-12 overflow-hidden">
            <div className="md:w-1/2">
              <span className="text-brand-blue/80 font-mono text-xs font-bold uppercase tracking-widest mb-3 block">Preview</span>
              <h3 className="text-3xl font-bold text-text-primary mb-4">Deep Learning + Econometrics</h3>
              <p className="text-text-secondary leading-relaxed mb-6">
                Our architecture handles non-stationary financial data by combining traditional GARCH volatility paths with modern attention-based Transformers, allowing you to see standard deviations before they are realized.
              </p>
              <ul className="space-y-3">
                {['Isolation Forests for structural anomalies', 'Autoencoders for compressed representations', 'LSTM/Transformers for multi-horizon forecasts'].map((item, i) => (
                  <li key={i} className="flex items-center gap-3 text-sm text-text-secondary font-medium">
                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="md:w-1/2 w-full aspect-[4/3] bg-page-bg rounded-xl border border-card-border p-4 relative overflow-hidden flex flex-col">
              <div className="flex justify-between items-center mb-4 text-xs font-mono text-text-secondary">
                 <span>USD/SPX • 1D</span>
                 <span className="text-brand-blue">Vol: High</span>
              </div>
              <div className="flex-1 w-full relative">
                 {/* Faux mini-chart SVG */}
                 <svg className="absolute inset-0 w-full h-full stroke-brand-blue" preserveAspectRatio="none" viewBox="0 0 100 100">
                    {/* Grid */}
                    <path d="M0,25 L100,25 M0,50 L100,50 M0,75 L100,75" stroke="#e2e8f0" strokeWidth="0.5" strokeDasharray="2 2" fill="none" />
                    
                    {/* Anomaly shaded region */}
                    <rect x="65" y="0" width="20" height="100" fill="rgba(34, 211, 238, 0.1)" stroke="none" />
                    <line x1="65" y1="0" x2="65" y2="100" stroke="rgba(34, 211, 238, 0.5)" strokeWidth="0.5" strokeDasharray="4 4" />
                    <line x1="85" y1="0" x2="85" y2="100" stroke="rgba(34, 211, 238, 0.5)" strokeWidth="0.5" strokeDasharray="4 4" />
                    
                    {/* Main line */}
                    <path d="M0,80 Q10,70 20,75 T40,60 T60,65 T70,30 T80,45 T100,20" fill="none" strokeWidth="1.5" />
                    <circle cx="70" cy="30" r="3" fill="#22d3ee" className="animate-pulse" />
                 </svg>
                 <div className="absolute top-[20px] right-[25%] bg-white border border-cyan-200 text-[10px] px-2 py-1 rounded shadow-lg text-cyan-700 font-bold font-mono">
                   Z-Score &gt; 3.0
                 </div>
              </div>
            </div>
          </div>
        </section>

        <HowItWorks />
        <FinalCTA />
      </main>

      {/* Footer */}
      <footer className="w-full py-8 text-center text-sm font-mono text-text-secondary border-t border-card-border bg-page-bg">
        <p className="mb-2">Market Anomaly Detection & Prediction</p>
        <p className="opacity-60">MIT License • {new Date().getFullYear()}</p>
      </footer>

      {/* Down Scroll Indicator */}
      <motion.div 
        style={{ opacity }}
        className="fixed bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center text-text-secondary pointer-events-none hidden md:flex"
      >
        <span className="text-[10px] uppercase font-bold tracking-widest mb-2 font-mono">Scroll</span>
        <div className="w-px h-12 bg-gradient-to-b from-text-secondary to-transparent" />
      </motion.div>
    </div>
  )
}
