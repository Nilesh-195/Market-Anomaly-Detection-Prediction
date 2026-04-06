import { motion } from 'framer-motion'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, ChevronLeft, ChevronRight,
  TrendingUp, History, Activity, Layers, AlertCircle, BarChart3, Settings, LifeBuoy
} from 'lucide-react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { path: '/app/dashboard', label: 'Dashboard', Icon: LayoutDashboard },
  { path: '/app/anomalies', label: 'Anomaly Detection', Icon: AlertCircle },
  { path: '/app/forecast', label: 'Forecast', Icon: TrendingUp },
  { path: '/app/evaluation', label: 'Evaluation', Icon: BarChart3 },
  { path: '/app/regime', label: 'Regime', Icon: Layers },
  { path: '/app/historical', label: 'Historical', Icon: History },
]

const BOTTOM_ITEMS = [
    { path: '/app/settings', label: 'Settings', Icon: Settings },
    { path: '/app/support', label: 'Support', Icon: LifeBuoy },
]

export default function Sidebar({ sidebarCollapsed, setSidebarCollapsed }) {
  return (
    <motion.aside
      animate={{ width: sidebarCollapsed ? '5rem' : '16rem' }}
      transition={{ duration: 0.3, ease: 'inOut' }}
      className="fixed left-0 top-0 bottom-0 z-30 bg-card-bg border-r border-card-border flex flex-col"
    >
        <div className="flex items-center justify-between h-16 border-b border-card-border px-4">
            <div className={clsx("font-bold text-sm leading-tight", { 'hidden': sidebarCollapsed })}>
              Market Anomaly Detection & Forecasting
            </div>
            <button
                onClick={() => setSidebarCollapsed(v => !v)}
                className="p-1 rounded-full text-text-secondary hover:bg-surface"
            >
                {sidebarCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            </button>
        </div>
      
      <nav className="flex-1 py-4 space-y-1 px-2">
        {NAV_ITEMS.map(({ path, label, Icon }) => (
          <NavLink
            key={path}
            to={path}
            className={({ isActive }) =>
              clsx(
                'w-full flex items-center gap-3 rounded-md px-3 py-2.5 transition-colors text-sm font-medium',
                { 'justify-center': sidebarCollapsed },
                isActive
                  ? 'bg-brand-blue text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface'
              )
            }
          >
            <Icon size={20} className="flex-shrink-0" />
            <span className={clsx({ 'hidden': sidebarCollapsed })}>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="py-4 space-y-1 px-2">
        {BOTTOM_ITEMS.map(({ path, label, Icon }) => (
            <NavLink
                key={path}
                to={path}
                className={({ isActive }) =>
                clsx(
                    'w-full flex items-center gap-3 rounded-md px-3 py-2.5 transition-colors text-sm font-medium',
                    { 'justify-center': sidebarCollapsed },
                    isActive
                    ? 'bg-brand-blue text-white'
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface'
                )
                }
            >
                <Icon size={20} className="flex-shrink-0" />
                <span className={clsx({ 'hidden': sidebarCollapsed })}>{label}</span>
            </NavLink>
        ))}
      </div>
    </motion.aside>
  )
}
