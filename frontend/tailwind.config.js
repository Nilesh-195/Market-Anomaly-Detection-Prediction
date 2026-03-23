/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'page-bg':      '#FFFFFF',
        'card-bg':      '#FAFBFC',
        'card-border':  '#E5E7EB',
        'surface':      '#F3F4F6',
        'hover':        '#F0F1F3',
        'brand-blue':   '#2563EB',
        'brand-blue-dim':'#1D4ED8',
        'risk-normal':  '#059669',
        'risk-elevated':'#D97706',
        'risk-high':    '#EA580C',
        'risk-extreme': '#DC2626',
        'chart-blue':   '#3B82F6',
        'chart-purple': '#8B5CF6',
        'chart-cyan':   '#0891B2',
        'chart-green':  '#059669',
        'chart-red':    '#DC2626',
        'text-primary': '#111827',
        'text-secondary':'#6B7280',
        'text-muted':   '#9CA3AF',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        xl: '12px',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'shimmer': 'shimmer 1.5s infinite',
        'spin-once': 'spin 0.6s linear',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}

