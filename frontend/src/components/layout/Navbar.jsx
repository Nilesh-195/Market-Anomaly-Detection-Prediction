import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Bell, User, CheckCircle } from 'lucide-react'
import clsx from 'clsx'
import { ASSETS } from '../../constants/config'

export default function Navbar({
  selectedAsset, onAssetChange,
  apiOnline,
}) {
  const [assetOpen, setAssetOpen] = useState(false)
  const currentAsset = ASSETS.find(a => a.ticker === selectedAsset) || ASSETS[0]

  return (
    <header className="fixed top-0 right-0 z-20 h-16 bg-card-bg border-b border-card-border flex items-center px-6 gap-6" 
            style={{ left: '16rem' }}>
      
      {/* Asset selector */}
      <div className="flex items-center gap-2">
        {ASSETS.map(asset => (
            <button 
                key={asset.ticker}
                onClick={() => onAssetChange(asset.ticker)}
                className={clsx(
                    "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                    selectedAsset === asset.ticker ? 'bg-surface text-text-primary' : 'text-text-secondary hover:bg-surface'
                )}
            >
                {asset.ticker}
            </button>
        ))}
      </div>

      {/* Right */}
      <div className="flex items-center gap-4 ml-auto">
        <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={20}/>
            <input 
                type="text"
                placeholder="Search Markets..."
                className="bg-surface border border-transparent rounded-lg pl-10 pr-4 py-2 text-sm w-64 focus:outline-none focus:ring-2 focus:ring-brand-blue"
            />
        </div>

        <div className={clsx(
          'flex items-center gap-2 text-xs font-mono font-medium px-2.5 py-1 rounded-md border',
          apiOnline
            ? 'text-green-600 border-green-200 bg-green-50'
            : 'text-red-600 border-red-200 bg-red-50'
        )}>
          <CheckCircle size={14} />
          API Active
        </div>

        <button className="p-2 rounded-full hover:bg-surface">
            <Bell size={20} className="text-text-secondary"/>
        </button>
        <button className="p-1 rounded-full hover:bg-surface">
            <User size={24} className="text-text-secondary"/>
        </button>
      </div>
    </header>
  )
}
