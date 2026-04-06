import React from 'react'
import { motion } from 'framer-motion'
import { Activity, TrendingDown, Layers, BrainCircuit, BarChart3, BellRing } from 'lucide-react'

const features = [
  {
    icon: Activity,
    title: "Anomaly Timeline",
    description: "Scan historical market action to pinpoint exact dates where volume or price behaved irregularly.",
    color: "from-blue-500 to-blue-600"
  },
  {
    icon: TrendingDown,
    title: "Regime Shifts",
    description: "Identify broader macro changes and volatility expansions using advanced sequential models.",
    color: "from-cyan-500 to-blue-500"
  },
  {
    icon: BarChart3,
    title: "Forecast Fan Chart",
    description: "Look into the future. Deep learning generates median paths with statistical confidence intervals.",
    color: "from-indigo-500 to-purple-600"
  },
  {
    icon: BrainCircuit,
    title: "Explainability",
    description: "Don't just get an alert. Understand exactly which features drove the anomaly score out of bounds.",
    color: "from-blue-600 to-indigo-600"
  },
  {
    icon: Layers,
    title: "Per-asset Evaluation",
    description: "Compare isolation forests, autoencoders, and GARCH models head-to-head for every single ticker.",
    color: "from-cyan-600 to-teal-500"
  },
  {
    icon: BellRing,
    title: "Actionable Insights",
    description: "FastAPI pulls near real-time daily data, scoring it against historical boundaries automatically.",
    color: "from-purple-500 to-pink-500"
  }
]

export default function FeatureGrid() {
  return (
    <section id="features" className="py-24 px-6 max-w-7xl mx-auto w-full">
      <div className="text-center mb-16 max-w-2xl mx-auto">
        <h2 className="text-3xl md:text-4xl font-bold text-text-primary mb-4 tracking-tight">
          Everything you need to map the market.
        </h2>
        <p className="text-text-secondary text-lg">
          We combine econometrics and machine learning to build a robust suite of analytical tools.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {features.map((feature, i) => (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.5, delay: i * 0.1 }}
            key={i}
            className="group p-6 rounded-2xl bg-white border border-card-border hover:shadow-xl hover:shadow-brand-blue/5 hover:-translate-y-1 transition-all duration-300 relative overflow-hidden"
          >
            {/* Subtle top gradient line */}
            <div className={`absolute top-0 left-0 w-full h-1 bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-100 transition-opacity`} />
            
            <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <feature.icon className="text-brand-blue" strokeWidth={1.5} size={24} />
            </div>
            
            <h3 className="text-xl font-bold text-text-primary mb-3">
              {feature.title}
            </h3>
            
            <p className="text-text-secondary leading-relaxed">
              {feature.description}
            </p>
          </motion.div>
        ))}
      </div>
    </section>
  )
}
