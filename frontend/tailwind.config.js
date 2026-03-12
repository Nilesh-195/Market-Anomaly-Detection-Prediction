/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'page-bg':      '#0A0F1E',
        'card-bg':      '#0D1424',
        'card-border':  '#1A2640',
        'surface':      '#111827',
        'hover':        '#162033',
        'brand-blue':   '#2563EB',
        'brand-blue-dim':'#1D4ED8',
        'risk-normal':  '#10B981',
        'risk-elevated':'#F59E0B',
        'risk-high':    '#F97316',
        'risk-extreme': '#EF4444',
        'chart-blue':   '#3B82F6',
        'chart-purple': '#8B5CF6',
        'chart-cyan':   '#06B6D4',
        'chart-green':  '#10B981',
        'chart-red':    '#EF4444',
        'text-primary': '#F1F5F9',
        'text-secondary':'#64748B',
        'text-muted':   '#334155',
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

