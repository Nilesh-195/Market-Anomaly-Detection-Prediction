import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Activity } from 'lucide-react'

export default function MinimalNav() {
  return (
    <motion.nav 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className="w-full fixed top-0 z-50 px-6 py-4 flex items-center justify-between border-b border-card-border bg-page-bg/80 backdrop-blur-lg"
    >
      <Link to="/" className="flex items-center gap-2 group">
        <div className="bg-brand-blue text-white p-1.5 rounded-md flex items-center justify-center transition-transform group-hover:scale-105">
          <Activity size={20} />
        </div>
        <span className="font-mono font-bold text-xl text-brand-blue tracking-tight">AnomalyIQ</span>
      </Link>
      
      <div className="hidden md:flex items-center gap-8 text-sm font-medium text-text-secondary">
        <a href="#features" className="hover:text-brand-blue transition-colors">Features</a>
        <a href="#how-it-works" className="hover:text-brand-blue transition-colors">How it works</a>
        <a href="#models" className="hover:text-brand-blue transition-colors">Models</a>
        <a href="https://github.com/your-username/Market_Anomaly_Detection_Prediction" target="_blank" rel="noreferrer" className="hover:text-brand-blue transition-colors">GitHub</a>
      </div>

      <div>
        <Link 
          to="/app/dashboard"
          className="bg-brand-blue text-white px-5 py-2.5 rounded-lg font-semibold text-sm hover:bg-brand-blue/90 transition-all hover:shadow-lg hover:shadow-brand-blue/20"
        >
          Start
        </Link>
      </div>
    </motion.nav>
  )
}
