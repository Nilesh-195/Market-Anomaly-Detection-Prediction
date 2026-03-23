import clsx from 'clsx'

export function Badge({ children, className = '', variant = 'default' }) {
  const variants = {
    default:  'bg-surface text-text-secondary',
    blue:     'bg-blue-50 text-brand-blue border border-brand-blue/20',
    green:    'bg-green-50 text-risk-normal border border-risk-normal/20',
    yellow:   'bg-yellow-50 text-risk-elevated border border-risk-elevated/20',
    orange:   'bg-orange-50 text-risk-high border border-risk-high/20',
    red:      'bg-red-50 text-risk-extreme border border-risk-extreme/20',
    purple:   'bg-purple-50 text-chart-purple border border-chart-purple/20',
    cyan:     'bg-cyan-50 text-chart-cyan border border-chart-cyan/20',
  }
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium font-mono',
        variants[variant] ?? variants.default,
        className
      )}
    >
      {children}
    </span>
  )
}
