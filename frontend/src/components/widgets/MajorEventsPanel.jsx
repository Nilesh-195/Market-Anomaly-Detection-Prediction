import { Card } from '../ui/Card'
import { Badge } from '../ui/Badge'
import { formatDate } from '../../utils/formatters'

function impactVariant(impact) {
  const normalized = String(impact || '').toLowerCase()
  if (normalized === 'extreme') return 'red'
  if (normalized === 'high') return 'orange'
  if (normalized === 'medium') return 'yellow'
  return 'default'
}

export default function MajorEventsPanel({ events = [], showMarkers, onToggleMarkers, loading = false, error = null }) {
  const rows = Array.isArray(events) ? [...events].sort((a, b) => String(b.date).localeCompare(String(a.date))) : []

  return (
    <Card className="p-4">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-[0.12em] text-text-primary">Major Events</h3>
          <p className="mt-1 text-xs text-text-secondary">Labeled crash events tied to this asset.</p>
        </div>
        <label className="inline-flex items-center gap-2 rounded-md border border-card-border bg-surface/30 px-2 py-1 text-[11px] font-semibold text-text-secondary">
          <input
            type="checkbox"
            checked={Boolean(showMarkers)}
            onChange={(e) => onToggleMarkers?.(e.target.checked)}
            className="rounded border-card-border text-brand-blue"
          />
          Show markers
        </label>
      </div>

      {error && (
        <div className="mb-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="h-[200px] animate-pulse rounded-lg bg-surface" />
      ) : !rows.length ? (
        <div className="rounded-lg border border-dashed border-card-border bg-surface/40 px-4 py-6 text-center text-sm text-text-muted">
          No major labeled events for this asset in the selected scope.
        </div>
      ) : (
        <div className="max-h-[240px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-card-border">
                {['Date', 'Event', 'Impact', 'Description'].map((heading) => (
                  <th key={heading} className="px-1.5 pb-1.5 text-left font-semibold uppercase tracking-[0.1em] text-text-secondary">
                    {heading}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((event) => (
                <tr key={`${event.date}-${event.event}`} className="border-b border-card-border/50 align-top">
                  <td className="px-1.5 py-1.5 font-mono text-text-secondary">{formatDate(event.date, 'MMM dd, yyyy')}</td>
                  <td className="px-1.5 py-1.5 font-semibold text-text-primary">{event.event}</td>
                  <td className="px-1.5 py-1.5">
                    <Badge variant={impactVariant(event.impact)}>{event.impact || 'unknown'}</Badge>
                  </td>
                  <td className="px-1.5 py-1.5 text-text-secondary">{event.description || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
