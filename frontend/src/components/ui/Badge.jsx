import clsx from 'clsx'

export function Badge({ children, className = '', variant = 'default' }) {
  const variants = {
    default:  'bg-surface text-[#64748B]',
    blue:     'bg-[#2563EB]/10 text-[#3B82F6] border border-[#2563EB]/20',
    green:    'bg-[#10B981]/10 text-[#10B981] border border-[#10B981]/20',
    yellow:   'bg-[#F59E0B]/10 text-[#F59E0B] border border-[#F59E0B]/20',
    orange:   'bg-[#F97316]/10 text-[#F97316] border border-[#F97316]/20',
    red:      'bg-[#EF4444]/10 text-[#EF4444] border border-[#EF4444]/20',
    purple:   'bg-[#8B5CF6]/10 text-[#8B5CF6] border border-[#8B5CF6]/20',
    cyan:     'bg-[#06B6D4]/10 text-[#06B6D4] border border-[#06B6D4]/20',
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
