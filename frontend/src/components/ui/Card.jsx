import clsx from 'clsx'

export function Card({ children, className = '', hover = false, ...props }) {
  return (
    <div
      className={clsx(
        'bg-card-bg border border-card-border rounded-xl p-5',
        hover && 'transition-all duration-200 hover:border-[#2563EB]/40 hover:scale-[1.005]',
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}
