import { Construction } from 'lucide-react'
import { Card } from '../components/ui/Card'

export default function PlaceholderPage({ title, description }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-text-primary mb-1">{title}</h1>
        <p className="text-text-secondary text-sm">{description}</p>
      </div>
      
      <Card className="flex flex-col items-center justify-center min-h-[400px] text-center p-8">
        <div className="w-16 h-16 rounded-full bg-blue-50 flex items-center justify-center text-brand-blue mb-4">
          <Construction size={32} />
        </div>
        <h2 className="text-xl font-semibold text-text-primary mb-2">Page Under Construction</h2>
        <p className="text-text-secondary max-w-md">
          This feature is part of the new redesign and is currently being built. It will include advanced visualization and dynamic interaction capabilities.
        </p>
      </Card>
    </div>
  )
}
