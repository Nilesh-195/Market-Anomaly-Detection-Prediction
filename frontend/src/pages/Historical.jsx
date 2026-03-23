import { useState } from 'react'
import {
  Calendar, AlertTriangle, TrendingDown, Activity,
  ChevronDown, ChevronUp, Filter, Loader2,
} from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Card } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import AnomalyTable from '../components/widgets/AnomalyTable'
import AnomalyCalendar from '../components/widgets/AnomalyCalendar'
import { formatDate, formatScore } from '../utils/formatters'
import { getRiskColor, getRiskLabel } from '../utils/riskHelpers'
import clsx from 'clsx'

export default function Historical({ historical, selectedAsset, loading, error }) {
  const [showCalendar, setShowCalendar] = useState(true)
  const [severityFilter, setSeverityFilter] = useState('all')

  const events = historical?.events ?? []
  const totalAnomalies = historical?.total_anomaly_days ?? 0
  const chartData = historical?.chart_data ?? []

  // Filter events by severity
  const filteredEvents = events.filter(ev => {
    if (severityFilter === 'all') return true
    const score = ev.ensemble_score ?? 0
    if (severityFilter === 'extreme') return score >= 75
    if (severityFilter === 'high') return score >= 60 && score < 75
    if (severityFilter === 'elevated') return score >= 40 && score < 60
    return score < 40
  })

  // Stats
  const extremeCount = events.filter(e => (e.ensemble_score ?? 0) >= 75).length
  const highCount = events.filter(e => (e.ensemble_score ?? 0) >= 60 && (e.ensemble_score ?? 0) < 75).length
  const avgScore = events.length > 0
    ? events.reduce((sum, e) => sum + (e.ensemble_score ?? 0), 0) / events.length
    : 0

  // Most recent anomaly
  const mostRecent = events[0]

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Historical Analysis</h1>
          <p className="text-text-secondary text-sm">Past anomaly events and market crash detection</p>
        </div>
        <div className="flex items-center justify-center h-[400px]">
          <Loader2 size={32} className="animate-spin text-brand-blue" />
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary mb-1">Historical Analysis</h1>
          <p className="text-text-secondary text-sm">
            {selectedAsset} anomaly events detected by ensemble model
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowCalendar(v => !v)}
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
              showCalendar
                ? 'bg-brand-blue text-white'
                : 'bg-surface border border-card-border text-text-secondary hover:text-text-primary'
            )}
          >
            <Calendar size={14} />
            Calendar View
          </button>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={16} className="text-risk-extreme" />
            <span className="text-text-secondary text-[11px] uppercase tracking-wider">Total Anomalies</span>
          </div>
          <div className="font-mono font-bold text-3xl text-text-primary">{totalAnomalies}</div>
          <div className="text-xs text-text-muted mt-1">events detected</div>
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown size={16} className="text-risk-extreme" />
            <span className="text-text-secondary text-[11px] uppercase tracking-wider">Extreme Events</span>
          </div>
          <div className="font-mono font-bold text-3xl text-risk-extreme">{extremeCount}</div>
          <div className="text-xs text-text-muted mt-1">score ≥ 75</div>
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-2">
            <Activity size={16} className="text-risk-high" />
            <span className="text-text-secondary text-[11px] uppercase tracking-wider">High Risk Events</span>
          </div>
          <div className="font-mono font-bold text-3xl text-risk-high">{highCount}</div>
          <div className="text-xs text-text-muted mt-1">score 60-75</div>
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-2">
            <Activity size={16} className="text-brand-blue" />
            <span className="text-text-secondary text-[11px] uppercase tracking-wider">Average Score</span>
          </div>
          <div className="font-mono font-bold text-3xl" style={{ color: getRiskColor(avgScore) }}>
            {formatScore(avgScore)}
          </div>
          <div className="text-xs text-text-muted mt-1">across all events</div>
        </Card>
      </div>

      {/* Most Recent Anomaly */}
      {mostRecent && (
        <Card className="bg-gradient-to-r from-red-50 to-orange-50 border-red-200">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-text-secondary text-xs uppercase tracking-wider mb-1">Most Recent Anomaly</div>
              <div className="flex items-center gap-3">
                <span className="font-mono text-lg text-text-primary">
                  {formatDate(mostRecent.date, 'MMMM dd, yyyy')}
                </span>
                <Badge variant={mostRecent.ensemble_score >= 75 ? 'red' : mostRecent.ensemble_score >= 60 ? 'orange' : 'yellow'}>
                  {getRiskLabel(mostRecent.ensemble_score)}
                </Badge>
              </div>
            </div>
            <div className="text-right">
              <div className="font-mono font-bold text-3xl" style={{ color: getRiskColor(mostRecent.ensemble_score) }}>
                {formatScore(mostRecent.ensemble_score)}
              </div>
              <div className="text-xs text-text-muted">ensemble score</div>
            </div>
          </div>
        </Card>
      )}

      {/* Calendar View */}
      <AnimatePresence>
        {showCalendar && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Card>
              <h3 className="text-lg font-semibold text-text-primary mb-4">Anomaly Calendar</h3>
              <AnomalyCalendar events={events} />
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Events Table */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold text-text-primary">Anomaly Events</h3>
            <p className="text-text-secondary text-sm">
              {filteredEvents.length} events {severityFilter !== 'all' && `(filtered by ${severityFilter})`}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-text-muted" />
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="bg-surface border border-card-border rounded-lg px-3 py-1.5 text-sm text-text-primary"
            >
              <option value="all">All Severities</option>
              <option value="extreme">Extreme (≥75)</option>
              <option value="high">High (60-75)</option>
              <option value="elevated">Elevated (40-60)</option>
              <option value="normal">Normal (&lt;40)</option>
            </select>
          </div>
        </div>
        <AnomalyTable events={filteredEvents} loading={loading} maxRows={20} />
      </Card>
    </div>
  )
}
