import { motion } from 'framer-motion'
import {
  DollarSign, Activity, Waves, AlertCircle,
} from 'lucide-react'
import { Card } from '../components/ui/Card'
import { SkeletonCard, SkeletonChart } from '../components/ui/Skeleton'
import KPICard from '../components/cards/KPICard'
import RiskGauge from '../components/widgets/RiskGauge'
import ModelConsensus from '../components/widgets/ModelConsensus'
import AlertBanner from '../components/widgets/AlertBanner'
import AnomalyTable from '../components/widgets/AnomalyTable'
import PriceAreaChart from '../components/charts/PriceAreaChart'
import RiskScoreChart from '../components/charts/RiskScoreChart'
import {
  formatPrice, formatScore, formatZScore, formatVolatility,
} from '../utils/formatters'
import { getRiskColor, getZScoreColor, getVolatilityColor } from '../utils/riskHelpers'

function sectionVariants(i = 0) {
  return {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0, transition: { delay: i * 0.08, duration: 0.35 } },
  }
}

export default function Dashboard({ current, historical, loading }) {
  const score    = current?.ensemble_score ?? 0
  const scores   = current?.model_scores   ?? {}
  const price    = current?.price
  const zscore   = current?.zscore
  const vol      = current?.volatility
  const anomCount = historical?.anomalies?.length ?? 0

  // Build time series for charts
  const chartData    = historical?.chart_data ?? []
  const anomalyPts   = (historical?.anomalies ?? []).slice(0, 20).map(a => ({
    date: a.date,
    close: a.price ?? null,
  }))

  return (
    <div className="space-y-4">
      {/* Alert banner */}
      <AlertBanner current={current} />

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <KPICard
          label="Current Price" index={0} loading={loading}
          value={price ? formatPrice(price) : '—'}
          icon={DollarSign}
          delta={current?.price_change_pct ? `${current.price_change_pct > 0 ? '+' : ''}${current.price_change_pct?.toFixed(2)}%` : null}
        />
        <KPICard
          label="Risk Score" index={1} loading={loading}
          value={formatScore(score)}
          valueColor={getRiskColor(score)}
          icon={Activity}
          delta={current?.score_delta != null ? `${current.score_delta > 0 ? '+' : ''}${formatScore(current.score_delta)}` : null}
        />
        <KPICard
          label="Z-Score" index={2} loading={loading}
          value={zscore != null ? formatZScore(zscore) : '—'}
          valueColor={zscore != null ? getZScoreColor(zscore) : undefined}
          icon={() => <span className="text-[#334155] text-sm">σ</span>}
        />
        <KPICard
          label="Volatility (30d)" index={3} loading={loading}
          value={vol != null ? formatVolatility(vol) : '—'}
          valueColor={vol != null ? getVolatilityColor(vol) : undefined}
          icon={Waves}
        />
        <KPICard
          label="Anomaly Count" index={4} loading={loading}
          value={String(anomCount)}
          icon={AlertCircle}
          delta={null}
          deltaLabel="in period"
        />
      </div>

      {/* Price chart + Gauge */}
      <motion.div
        variants={sectionVariants(1)} initial="initial" animate="animate"
        className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4"
      >
        <Card>
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-[#F1F5F9] font-medium">Price Chart</div>
              <div className="text-[#64748B] text-xs">With anomaly markers</div>
            </div>
          </div>
          {loading
            ? <div className="h-[220px] bg-surface rounded-lg animate-pulse" />
            : <PriceAreaChart data={chartData} anomalyPoints={anomalyPts} />
          }
        </Card>

        <Card className="flex flex-col items-center justify-center">
          <div className="text-[#F1F5F9] font-medium mb-1 self-start">Risk Gauge</div>
          <div className="text-[#64748B] text-xs mb-4 self-start">Ensemble score</div>
          {loading
            ? <div className="h-[200px] w-full bg-surface rounded-lg animate-pulse" />
            : <RiskGauge score={score} />
          }
        </Card>
      </motion.div>

      {/* Risk timeline */}
      <motion.div variants={sectionVariants(2)} initial="initial" animate="animate">
        <Card>
          <div className="text-[#F1F5F9] font-medium mb-1">Risk Score Timeline</div>
          <div className="text-[#64748B] text-xs mb-4">Ensemble anomaly score over selected period</div>
          {loading
            ? <div className="h-[160px] bg-surface rounded-lg animate-pulse" />
            : <RiskScoreChart data={chartData} />
          }
        </Card>
      </motion.div>

      {/* Events table + Model consensus */}
      <motion.div
        variants={sectionVariants(3)} initial="initial" animate="animate"
        className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4"
      >
        <Card>
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-[#F1F5F9] font-medium">Detected Anomaly Events</div>
              <div className="text-[#64748B] text-xs">{anomCount} events in selected period</div>
            </div>
          </div>
          <AnomalyTable
            events={historical?.anomalies ?? []}
            loading={loading}
            maxRows={8}
          />
        </Card>

        <ModelConsensus current={current} />
      </motion.div>
    </div>
  )
}
