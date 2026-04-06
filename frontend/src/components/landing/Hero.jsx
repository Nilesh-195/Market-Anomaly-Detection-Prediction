import React, { Suspense, useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion, useReducedMotion } from 'framer-motion'
import { ArrowRight, ActivitySquare, ShieldAlert, LineChart } from 'lucide-react'
import MarketPulseFallback from './MarketPulseFallback'

// Lazy load the 3D sculpture to keep initial load fast
const MarketPulseSculpture = React.lazy(() => import('./MarketPulseSculpture'))

export default function Hero() {
  const shouldReduceMotion = useReducedMotion()
  const [use3D, setUse3D] = useState(false)

  // Only attempt to load 3D if not respecting reduced motion, and delayed to avoid blocking
  useEffect(() => {
    if (!shouldReduceMotion) {
      const timer = setTimeout(() => setUse3D(true), 500)
      return () => clearTimeout(timer)
    }
  }, [shouldReduceMotion])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { 
        staggerChildren: 0.15,
        delayChildren: 0.2
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, y: 0, 
      transition: { duration: 0.7, ease: [0.22, 1, 0.36, 1] } 
    }
  }

  return (
    <section className="relative pt-32 pb-20 md:pt-48 md:pb-32 px-6 overflow-hidden max-w-7xl mx-auto w-full">
      {/* Background radial gradient blobs strictly desktop */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none hidden md:block">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-100/50 rounded-full blur-[100px]" />
        <div className="absolute top-[20%] right-[-10%] w-[40%] h-[40%] bg-cyan-100/40 rounded-full blur-[100px]" />
      </div>

      <div className="w-full grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
        
        {/* Left Column Text */}
        <motion.div 
          className="lg:col-span-6 flex flex-col items-start text-left"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div variants={itemVariants} className="mb-6 inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-50 border border-blue-100 text-xs font-semibold text-brand-blue uppercase tracking-wider">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-blue animate-pulse" />
            Market Intelligence • Detection • Forecasting
          </motion.div>

          <motion.h1 
            variants={itemVariants}
            className="text-4xl md:text-6xl font-bold tracking-tight text-text-primary leading-[1.1] mb-6"
          >
            See market anomalies <br className="hidden md:block"/> 
            before they become <span className="text-brand-blue relative whitespace-nowrap">
              headlines.
              <svg className="absolute bottom-1 left-0 w-full h-3 -z-10 text-blue-200" viewBox="0 0 100 20" preserveAspectRatio="none">
                <path d="M0 15 Q 50 0 100 15" fill="none" stroke="currentColor" strokeWidth="8" strokeLinecap="round" />
              </svg>
            </span>
          </motion.h1>

          <motion.p variants={itemVariants} className="text-lg md:text-xl text-text-secondary leading-relaxed mb-8 max-w-xl">
            A multi-asset dashboard that highlights unusual market behavior, explains why it matters, and visualizes forecasts with confidence.
          </motion.p>

          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto mb-10">
            <Link 
              to="/app/dashboard"
              className="inline-flex justify-center items-center gap-2 px-6 py-3.5 bg-brand-blue text-white rounded-xl font-semibold hover:bg-brand-blue/90 hover:-translate-y-0.5 hover:shadow-xl hover:shadow-brand-blue/20 transition-all"
            >
              Start Experience
              <ArrowRight size={18} />
            </Link>
            <Link 
              to="/app/anomalies"
              className="inline-flex justify-center items-center gap-2 px-6 py-3.5 bg-white text-text-primary border border-card-border rounded-xl font-semibold hover:bg-gray-50 hover:-translate-y-0.5 hover:shadow-md transition-all"
            >
              Explore Anomalies
            </Link>
          </motion.div>

          {/* Trust Strip */}
          <motion.div variants={itemVariants} className="flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-text-secondary font-medium">
            <div className="flex items-center gap-1.5"><ActivitySquare size={16} className="text-blue-500"/> FastAPI backend</div>
            <div className="flex items-center gap-1.5"><LineChart size={16} className="text-cyan-500"/> Interactive charts</div>
            <div className="flex items-center gap-1.5"><ShieldAlert size={16} className="text-purple-500"/> Ensemble scoring</div>
          </motion.div>

          <motion.div variants={itemVariants} className="mt-8 flex gap-2 flex-wrap text-xs font-mono text-text-secondary">
             <span className="px-2 py-1 bg-white border border-card-border rounded">Assets: S&P500, VIX, TSLA...</span>
             <span className="px-2 py-1 bg-white border border-card-border rounded">Signal: Explainable</span>
          </motion.div>

        </motion.div>

        {/* Right Column 3D / Illustration */}
        <motion.div 
          className="lg:col-span-6 w-full relative z-10"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.3 }}
        >
          <div className="relative w-full aspect-[4/3] rounded-2xl bg-white/50 border border-card-border backdrop-blur-xl shadow-2xl p-2 group">
            {use3D ? (
              <Suspense fallback={<MarketPulseFallback />}>
                <MarketPulseSculpture />
              </Suspense>
            ) : (
              <MarketPulseFallback />
            )}
            
            {/* Subtle floating decorative elements */}
            <div className="absolute -top-6 -right-6 w-24 h-24 bg-gradient-to-br from-cyan-400/20 to-blue-600/20 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-1000"></div>
          </div>
        </motion.div>

      </div>
    </section>
  )
}
