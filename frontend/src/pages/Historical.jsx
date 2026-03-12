import { motion } from 'framer-motion'
import { Card } from '../components/ui/Card'
import PriceAreaChart from '../components/charts/PriceAreaChart'
import DrawdownChart from '../components/charts/DrawdownChart'
import VolatilityChart from '../components/charts/VolatilityChart'
import AnomalyTable from '../components/widgets/AnomalyTable'
import AnomalyCalendar from '../components/widgets/AnomalyCalendar'

export default function Historical({ historical, loading }) {
  const chartData  = historical?.chart_data  ?? []
  const anomalies  = historical?.anomalies   ?? []
  const anomalyPts = anomalies.slice(0, 30).map(a => ({ date: a.date, close: a.price ?? null }))

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-[#F1F5F9] text-xl font-semibold">Historical Analysis</h1>
        <p className="text-[#64748B] text-sm mt-0.5">
          Full anomaly history · {anomalies.length} events detected
        </p>
      </div>

      {/* Price chart */}
      <Card>
        <div className="text-[#F1F5F9] font-medium mb-1">Price History with Anomaly Markers</div>
        <div className="text-[#64748B] text-xs mb-4">Red markers indicate anomaly events</div>
        {loading
          ? <div className="h-[240px] bg-surface rounded-lg animate-pulse" />
          : <PriceAreaChart data={chartData} anomalyPoints={anomalyPts} />
        }
      </Card>

      {/* Calendar heatmap */}
      <Card>
        <div className="text-[#F1F5F9] font-medium mb-1">Anomaly Heatmap</div>
        <div className="text-[#64748B] text-xs mb-4">Daily anomaly intensity — last 2 years</div>
        {loading
          ? <div className="h-[120px] bg-surface rounded-lg animate-pulse" />
          : <AnomalyCalendar events={anomalies} />
        }
      </Card>

      {/* Drawdown + Volatility */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <div className="text-[#F1F5F9] font-medium mb-1">Drawdown from Peak</div>
          <div className="text-[#64748B] text-xs mb-4">Rolling 252-day max drawdown</div>
          {loading
            ? <div className="h-[200px] bg-surface rounded-lg animate-pulse" />
            : <DrawdownChart data={chartData} />
          }
        </Card>
        <Card>
          <div className="text-[#F1F5F9] font-medium mb-1">Volatility Regime</div>
          <div className="text-[#64748B] text-xs mb-4">30-day rolling annualised volatility</div>
          {loading
            ? <div className="h-[200px] bg-surface rounded-lg animate-pulse" />
            : <VolatilityChart data={chartData} />
          }
        </Card>
      </div>

      {/* Full anomaly table */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <div>
            <div className="text-[#F1F5F9] font-medium">All Anomaly Events</div>
            <div className="text-[#64748B] text-xs">{anomalies.length} total events</div>
          </div>
        </div>
        <AnomalyTable events={anomalies} loading={loading} maxRows={anomalies.length} />
      </Card>
    </div>
  )
}
