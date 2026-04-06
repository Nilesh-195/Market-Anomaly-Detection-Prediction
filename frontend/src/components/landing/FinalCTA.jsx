import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Github } from 'lucide-react'

export default function FinalCTA() {
  return (
    <section className="py-24 px-6 max-w-7xl mx-auto w-full">
      <div className="bg-brand-blue rounded-3xl p-10 md:p-16 text-center relative overflow-hidden shadow-2xl shadow-brand-blue/20">
        
        {/* Decorative BG */}
        <div className="absolute top-0 right-0 -mr-20 -mt-20 w-64 h-64 bg-white/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-64 h-64 bg-cyan-400/20 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl mx-auto">
          <h2 className="text-3xl md:text-5xl font-bold text-white mb-6">
            Ready to explore the next anomaly?
          </h2>
          <p className="text-blue-100 text-lg mb-10 max-w-xl mx-auto">
            Experience the full dashboard. Interact with models, review regime shifts, and analyze historical crashes in real-time.
          </p>
          
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link 
              to="/app/dashboard"
              className="inline-flex justify-center items-center gap-2 px-8 py-4 bg-white text-brand-blue rounded-xl font-bold hover:bg-gray-50 hover:scale-105 transition-transform"
            >
              Start Dashboard
              <ArrowRight size={20} />
            </Link>
            <a 
              href="https://github.com/your-username/Market_Anomaly_Detection_Prediction" 
              target="_blank" 
              rel="noreferrer"
              className="inline-flex justify-center items-center gap-2 px-8 py-4 bg-brand-blue-dim border border-white/20 text-white rounded-xl font-bold hover:bg-brand-blue-light transition-colors"
            >
              <Github size={20} />
              View Source
            </a>
          </div>
          
          <p className="mt-8 text-xs text-blue-200/60 font-mono">
             Educational demo; not financial advice.
          </p>
        </div>
      </div>
    </section>
  )
}
