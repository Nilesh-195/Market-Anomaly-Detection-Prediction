/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'page-bg':      '#F7F8FA',  // Lighter gray background
        'card-bg':      '#FFFFFF',
        'card-border':  '#EAEBEE',  // Softer border
        'surface':      '#F0F2F5',
        'surface-alt':  '#E9EDF0',
        'hover':        '#F5F7F9',
        'brand-blue':   '#2563EB',  // A more vibrant blue
        'brand-blue-dim':'#1D4ED8',
        'risk-normal':  '#10B981',
        'risk-elevated':'#F59E0B',
        'risk-high':    '#EF4444',
        'risk-extreme': '#B91C1C',
        'chart-blue':   '#3B82F6',
        'chart-purple': '#8B5CF6',
        'chart-cyan':   '#06B6D4',
        'chart-green':  '#10B981',
        'chart-red':    '#EF4444',
        'text-primary': '#111827',  // Darker for better contrast
        'text-secondary':'#6B7280',
        'text-muted':   '#9CA3AF',
      },
      fontFamily: {
        mono: ['IBM Plex Mono', 'SFMono-Regular', 'monospace'],
        sans: ['Inter', 'Manrope', 'system-ui', 'sans-serif'], // Using Inter as primary
      },
      boxShadow: {
        'card': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'card-hover': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        'lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
      },
      borderRadius: {
        lg: '0.5rem',
        xl: '0.75rem',
        '2xl': '1rem',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'shimmer': 'shimmer 1.5s infinite',
        'spin-once': 'spin 0.6s linear',
        'float-up': 'floatUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards',
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        floatUp: {
          '0%': { opacity: 0, transform: 'translateY(10px)' },
          '100%': { opacity: 1, transform: 'translateY(0)' },
        }
      },
    },
  },
  plugins: [],
}

