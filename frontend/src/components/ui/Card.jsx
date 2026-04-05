import clsx from 'clsx'

export function Card({ children, className = '', hover = false, ...props }) {
  return (
    <div
      className={clsx(
        'relative overflow-hidden bg-card-bg border border-card-border rounded-2xl p-5 shadow-glass transition-all duration-300',
        hover && 'hover:border-brand-blue/35 hover:shadow-float hover:-translate-y-0.5',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
