import { format, parseISO, isValid } from 'date-fns'

export function formatPrice(value) {
  if (value == null) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

export function formatPct(value, decimals = 1) {
  if (value == null) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${Number(value).toFixed(decimals)}%`
}

export function formatScore(value) {
  if (value == null) return '—'
  return Number(value).toFixed(1)
}

export function formatZScore(value) {
  if (value == null) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${Number(value).toFixed(2)}σ`
}

export function formatDate(value, fmt = 'MMM dd') {
  if (!value) return '—'
  try {
    const d = typeof value === 'string' ? parseISO(value) : value
    return isValid(d) ? format(d, fmt) : String(value).slice(0, 10)
  } catch {
    return String(value).slice(0, 10)
  }
}

export function formatDateLong(value) {
  return formatDate(value, 'MMM dd, yyyy')
}

export function formatTime(date = new Date()) {
  return format(date, 'HH:mm:ss')
}

export function formatVolatility(value) {
  if (value == null) return '—'
  return `${Number(value * 100).toFixed(1)}%`
}
