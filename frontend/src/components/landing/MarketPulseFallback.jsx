import React from 'react'
import { motion } from 'framer-motion'
import { Activity } from 'lucide-react'

export default function MarketPulseFallback() {
  return (
    <div className="w-full h-full min-h-[400px] flex items-center justify-center bg-card-bg/50 rounded-2xl border border-card-border shadow-sm">
      <div className="relative w-full max-w-sm aspect-square p-8 flex flex-col items-center justify-center text-center gap-6">
        {/* Decorative background grid elements */}
        <div className="absolute inset-0 grid grid-cols-6 grid-rows-6 opacity-5">
          {Array.from({ length: 36 }).map((_, i) => (
            <div key={i} className="border-[0.5px] border-brand-blue" />
          ))}
        </div>
        
        <div className="relative z-10 w-24 h-24 bg-blue-50/50 rounded-3xl flex items-center justify-center border border-blue-100 shadow-xl shadow-brand-blue/5">
           <Activity size={48} className="text-brand-blue" strokeWidth={1.5} />
        </div>
        
        <div className="relative z-10">
          <h3 className="text-lg font-semibold text-brand-blue mb-2">Market Pulse</h3>
          <p className="text-sm text-text-secondary leading-relaxed">
            Continuously analyzing multi-asset data arrays to detect hidden volatility signals.
          </p>
        </div>
        
        {/* Simulated static waves */}
        <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-1.5 opacity-20">
          {[4, 8, 12, 16, 20, 16, 12, 8, 4].map((h, i) => (
            <div key={i} className="w-1.5 bg-brand-blue rounded-t-sm" style={{ height: `${h}px` }} />
          ))}
        </div>
      </div>
    </div>
  )
}
