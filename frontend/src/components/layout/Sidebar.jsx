import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  LayoutDashboard, ChevronLeft, ChevronRight,
  TrendingUp, History, Activity, Info, Layers, AlertCircle,
} from 'lucide-react'
import clsx from 'clsx'
import { getRiskColor, getRiskLabel } from '../../utils/riskHelpers'
import { formatScore } from '../../utils/formatters'

const NAV_ITEMS = [
  { id: 'dashboard',  label: 'Dashboard',   Icon: LayoutDashboard },
  { id: 'forecast',   label: 'Forecast',    Icon: TrendingUp },
  { id: 'historical', label: 'Historical',  Icon: History },
  { id: 'regime',     label: 'Regime',      Icon: Layers },
  { id: 'advanced-anomaly', label: 'Advanced',   Icon: AlertCircle },
]

export default function Sidebar({ activePage, onPageChange, sp500Score }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <motion.aside
      animate={{ width: collapsed ? 56 : 220 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="fixed left-0 top-16 bottom-0 z-30 bg-white border-r border-card-border flex flex-col overflow-hidden shadow-sm"
    >
      {/* Toggle */}
      <button
        onClick={() => setCollapsed(v => !v)}
        className="absolute -right-3 top-6 w-6 h-6 bg-white border border-card-border rounded-full flex items-center justify-center text-text-secondary hover:text-text-primary z-10"
      >
        {collapsed ? <ChevronRight size={12} /> : <ChevronLeft size={12} />}
      </button>

      {/* Nav items */}
      <nav className="flex-1 py-4 space-y-0.5">
        {NAV_ITEMS.map(({ id, label, Icon }) => {
          const active = activePage === id
          return (
            <button
              key={id}
              onClick={() => onPageChange(id)}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 transition-all text-sm relative',
                active
                  ? 'bg-blue-50 text-text-primary before:absolute before:left-0 before:top-0 before:bottom-0 before:w-[3px] before:bg-brand-blue before:rounded-r-full'
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface'
              )}
            >
              <Icon size={17} className="flex-shrink-0" />
              {!collapsed && (
                <span className="font-medium truncate">{label}</span>
              )}
            </button>
          )
        })}
      </nav>

      {/* S&P 500 mini risk widget */}
      {!collapsed && sp500Score != null && (
        <div className="m-3 p-3 bg-surface border border-card-border rounded-xl">
          <div className="text-text-secondary text-[10px] uppercase tracking-wider mb-1">S&P 500 Risk</div>
          <div
            className="font-mono font-bold text-2xl leading-none mb-0.5"
            style={{ color: getRiskColor(sp500Score) }}
          >
            {formatScore(sp500Score)}
          </div>
          <div
            className="text-xs font-medium"
            style={{ color: getRiskColor(sp500Score) }}
          >
            {getRiskLabel(sp500Score)}
          </div>
        </div>
      )}
    </motion.aside>
  )
}
