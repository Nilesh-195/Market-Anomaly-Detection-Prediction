import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, AlertCircle, X } from 'lucide-react'

export function Toast({ message, type = 'success', onClose }) {
  const icons = { success: CheckCircle, error: AlertCircle }
  const colours = {
    success: 'border-risk-normal text-risk-normal bg-green-50',
    error:   'border-risk-extreme text-risk-extreme bg-red-50',
  }
  const Icon = icons[type] || CheckCircle

  return (
    <motion.div
      initial={{ opacity: 0, y: -20, x: 0 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className={`fixed top-20 right-6 z-50 flex items-center gap-3 border ${colours[type]} rounded-xl px-4 py-3 shadow-lg min-w-[260px]`}
    >
      <Icon size={16} />
      <span className="text-sm text-text-primary flex-1">{message}</span>
      <button onClick={onClose} className="text-text-secondary hover:text-text-primary">
        <X size={14} />
      </button>
    </motion.div>
  )
}

export function ToastContainer({ toasts, removeToast }) {
  return (
    <AnimatePresence>
      {toasts.map((t) => (
        <Toast key={t.id} {...t} onClose={() => removeToast(t.id)} />
      ))}
    </AnimatePresence>
  )
}
