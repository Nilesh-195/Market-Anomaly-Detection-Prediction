import clsx from 'clsx'

export function Card({ children, className = '', hover = false, ...props }) {
  return (
    <div
      className={clsx(
        'bg-card-bg border border-card-border rounded-xl p-5 shadow-sm',
        hover && 'transition-all duration-200 hover:border-brand-blue/40 hover:shadow-md',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
