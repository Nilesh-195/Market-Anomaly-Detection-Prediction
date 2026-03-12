import { useEffect, useRef } from 'react'
import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import { getRiskColor, getRiskLabel } from '../../utils/riskHelpers'
import { COLOURS } from '../../constants/colours'

const ZONES = [
  { label: 'NORMAL',   range: '0-40',   color: COLOURS.riskNormal,   max: 40  },
  { label: 'ELEVATED', range: '40-60',  color: COLOURS.riskElevated, max: 60  },
  { label: 'HIGH RISK',range: '60-75',  color: COLOURS.riskHigh,     max: 75  },
  { label: 'EXTREME',  range: '75-100', color: COLOURS.riskExtreme,  max: 100 },
]

function Arc({ cx, cy, r, startAngle, endAngle, color, opacity = 1 }) {
  function polarToXY(angle) {
    const rad = (angle - 90) * Math.PI / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }
  const s = polarToXY(startAngle)
  const e = polarToXY(endAngle)
  const large = endAngle - startAngle > 180 ? 1 : 0
  return (
    <path
      d={`M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`}
      stroke={color} strokeWidth={8} strokeLinecap="round"
      fill="none" opacity={opacity}
    />
  )
}

export default function RiskGauge({ score = 0 }) {
  const displayScore = useMotionValue(0)
  const rounded = useTransform(displayScore, v => Math.round(v))

  useEffect(() => {
    const ctrl = animate(displayScore, score, { duration: 1, ease: 'easeOut' })
    return ctrl.stop
  }, [score])

  const color = getRiskColor(score)
  const label = getRiskLabel(score)

  // Arc: -135° to +135° = 270° total sweep
  const totalSweep = 270
  const startAngle = -135
  const cx = 100, cy = 100, r = 72

  const needleAngle = startAngle + (score / 100) * totalSweep
  const needleRad = (needleAngle - 90) * Math.PI / 180
  const needleLen = 52
  const nx = cx + needleLen * Math.cos(needleRad)
  const ny = cy + needleLen * Math.sin(needleRad)

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 180" className="w-full max-w-[220px]">
        {/* Background arc */}
        <Arc cx={cx} cy={cy} r={r} startAngle={-135} endAngle={135}
          color={COLOURS.surface} opacity={1} />

        {/* Zone arcs */}
        {ZONES.map((z, i) => {
          const prev = ZONES[i - 1]?.max ?? 0
          const s = startAngle + (prev / 100) * totalSweep
          const e = startAngle + (z.max / 100) * totalSweep
          const active = score >= prev && score < z.max
          const zoneActive = score >= prev
          return (
            <Arc key={z.label} cx={cx} cy={cy} r={r}
              startAngle={s} endAngle={Math.min(e, startAngle + (score / 100) * totalSweep)}
              color={z.color} opacity={zoneActive ? 1 : 0.15}
            />
          )
        })}

        {/* Needle */}
        <motion.line
          x1={cx} y1={cy} x2={nx} y2={ny}
          stroke={color} strokeWidth={2.5} strokeLinecap="round"
          initial={{ rotate: startAngle, originX: `${cx}px`, originY: `${cy}px` }}
          animate={{ rotate: needleAngle }}
          transition={{ duration: 1, ease: 'easeOut' }}
        />
        <circle cx={cx} cy={cy} r={5} fill={color} />

        {/* Score display */}
        <text x={cx} y={cy + 30} textAnchor="middle"
          fill={color} fontSize={28} fontFamily="monospace" fontWeight="bold">
          {Math.round(score)}
        </text>
        <text x={cx} y={cy + 46} textAnchor="middle"
          fill={color} fontSize={10} fontFamily="sans-serif" fontWeight="500">
          {label.toUpperCase()}
        </text>
      </svg>

      {/* Zone legend */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 mt-1 w-full max-w-[200px]">
        {ZONES.map(z => {
          const active = getRiskLabel(score) === (z.label === 'ELEVATED' ? 'Elevated' : z.label === 'HIGH RISK' ? 'High Risk' : z.label === 'EXTREME' ? 'Extreme' : 'Normal')
          return (
            <div key={z.label} className="flex items-center gap-1.5" style={{ opacity: active ? 1 : 0.3 }}>
              <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: z.color }} />
              <span className="text-[10px] font-mono text-[#64748B]">{z.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
