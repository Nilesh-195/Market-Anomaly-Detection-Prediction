export function SkeletonRow() {
  return (
    <div className="flex gap-4 py-2.5 px-2">
      <div className="h-4 bg-surface rounded w-24 flex-shrink-0" />
      <div className="h-4 bg-surface rounded w-20 flex-shrink-0" />
      <div className="h-4 bg-surface rounded w-20 flex-shrink-0" />
      <div className="h-4 bg-surface rounded w-24 flex-1" />
    </div>
  )
}
