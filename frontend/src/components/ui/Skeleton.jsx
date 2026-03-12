import clsx from 'clsx'

function SkeletonBlock({ className = '' }) {
  return (
    <div
      className={clsx(
        'rounded-lg bg-[#111827] relative overflow-hidden',
        className
      )}
    >
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[#1A2640]/60 to-transparent animate-[shimmer_1.5s_infinite] bg-[length:200%_100%]" />
    </div>
  )
}

export function SkeletonCard() {
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-5 space-y-3">
      <SkeletonBlock className="h-3 w-24" />
      <SkeletonBlock className="h-8 w-32" />
      <SkeletonBlock className="h-3 w-16" />
    </div>
  )
}

export function SkeletonChart({ height = 200 }) {
  return (
    <div className="bg-card-bg border border-card-border rounded-xl p-5">
      <SkeletonBlock className="h-4 w-36 mb-4" />
      <SkeletonBlock style={{ height }} />
    </div>
  )
}

export function SkeletonRow() {
  return (
    <div className="flex gap-4 px-4 py-3 border-b border-card-border">
      <SkeletonBlock className="h-3 w-20" />
      <SkeletonBlock className="h-3 w-16" />
      <SkeletonBlock className="h-3 w-12" />
      <SkeletonBlock className="h-3 w-32" />
      <SkeletonBlock className="h-3 w-20" />
    </div>
  )
}

export { SkeletonBlock as Skeleton }
