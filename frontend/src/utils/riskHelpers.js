import { COLOURS } from '../constants/colours'
import { RISK_THRESHOLDS } from '../constants/config'

export function getRiskLabel(score) {
  if (score == null) return 'Unknown'
  if (score < RISK_THRESHOLDS.normal)   return 'Normal'
  if (score < RISK_THRESHOLDS.elevated) return 'Elevated'
  if (score < RISK_THRESHOLDS.high)     return 'High Risk'
  return 'Extreme'
}

export function getRiskColor(score) {
  if (score == null) return COLOURS.textSecondary
  if (score < RISK_THRESHOLDS.normal)   return COLOURS.riskNormal
  if (score < RISK_THRESHOLDS.elevated) return COLOURS.riskElevated
  if (score < RISK_THRESHOLDS.high)     return COLOURS.riskHigh
  return COLOURS.riskExtreme
}

export function getRiskTailwind(score) {
  if (score == null) return 'text-[#64748B]'
  if (score < RISK_THRESHOLDS.normal)   return 'text-risk-normal'
  if (score < RISK_THRESHOLDS.elevated) return 'text-risk-elevated'
  if (score < RISK_THRESHOLDS.high)     return 'text-risk-high'
  return 'text-risk-extreme'
}

export function getRiskBgTailwind(score) {
  if (score == null) return 'bg-[#334155]/20 text-[#64748B]'
  if (score < RISK_THRESHOLDS.normal)
    return 'bg-risk-normal/10 text-risk-normal'
  if (score < RISK_THRESHOLDS.elevated)
    return 'bg-risk-elevated/10 text-risk-elevated'
  if (score < RISK_THRESHOLDS.high)
    return 'bg-risk-high/10 text-risk-high'
  return 'bg-risk-extreme/10 text-risk-extreme'
}

export function getRiskBorderTailwind(score) {
  if (score == null) return 'border-[#334155]'
  if (score < RISK_THRESHOLDS.normal)   return 'border-risk-normal'
  if (score < RISK_THRESHOLDS.elevated) return 'border-risk-elevated'
  if (score < RISK_THRESHOLDS.high)     return 'border-risk-high'
  return 'border-risk-extreme'
}

export function isAnomaly(score) {
  return score != null && score >= RISK_THRESHOLDS.elevated
}

export function getZScoreColor(z) {
  const abs = Math.abs(z)
  if (abs < 2) return COLOURS.riskNormal
  if (abs < 3) return COLOURS.riskElevated
  return COLOURS.riskExtreme
}

export function getVolatilityColor(vol) {
  // vol is a decimal (e.g. 0.15 = 15%)
  if (vol < 0.15) return COLOURS.riskNormal
  if (vol < 0.25) return COLOURS.riskElevated
  return COLOURS.riskExtreme
}
