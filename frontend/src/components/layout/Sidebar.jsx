import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  LayoutDashboard, History, TrendingUp, BarChart2,
  Settings, ChevronLeft, ChevronRight,
} from 'lucide-react'
import clsx from 'clsx'
import { getRiskColor, getRiskLabel } from '../../utils/riskHelpers'
import { formatScore } from '../../utils/formatters'

const NAV_ITEMS = [
  { id: 'dashboard',  label: 'Dashboard',    Icon: LayoutDashboard },
  { id: 'historical', label: 'Historical',   Icon: History         },
  { id: 'forecast',   label: 'Forecast',     Icon: TrendingUp      },
  { id: 'models',     label: 'Model Stats',  Icon: BarChart2       },
  { id: 'settings',   label: 'Settings',     Icon: Settings        },
]

export default function Sidebar({ activePage, onPageChange, sp500Score }) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <motion.aside
      animate={{ width: collapsed ? 56 : 220 }}
      transition={{ duration: 0.2, ease: 'easeInOut' }}
      className="fixed left-0 top-16 bottom-0 z-30 bg-card-bg border-r border-card-border flex flex-col overflow-hidden"
    >
      {/* Toggle */}
      <button
        onClick={() => setCollapsed(v => !v)}
        className="absolute -right-3 top-6 w-6 h-6 bg-card-bg border border-card-border rounded-full flex items-center justify-center text-[#64748B] hover:text-[#F1F5F9] z-10"
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
                  ? 'bg-surface text-[#F1F5F9] before:absolute before:left-0 before:top-0 before:bottom-0 before:w-[3px] before:bg-brand-blue before:rounded-r-full'
                  : 'text-[#64748B] hover:text-[#F1F5F9] hover:bg-surface/50'
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
          <div className="text-[#64748B] text-[10px] uppercase tracking-wider mb-1">S&P 500 Risk</div>
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
