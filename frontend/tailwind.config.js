/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'page-bg':      '#F2F6FB',  /** premium light backdrop */
        'card-bg':      '#FFFFFF',  /** crisp white cards */
        'card-border':  '#DBE4EF',  /** cool slate border */
        'surface':      '#EDF2F8',  /** elevated light surface */
        'surface-alt':  '#E7EEF7',
        'hover':        '#EAF1F9',
        'brand-blue':   '#0B3A63',  /** deep financial navy */
        'brand-blue-dim':'#1A5F8E',
        'risk-normal':  '#10B981',  /** Modern Emerald */
        'risk-elevated':'#F59E0B',  /** Amber */
        'risk-high':    '#EF4444',  /** Red */
        'risk-extreme': '#B91C1C',  /** Deep Red */
        'chart-blue':   '#1D6FDC',
        'chart-purple': '#5B7FE4',
        'chart-cyan':   '#159EC0',
        'chart-green':  '#10B981',
        'chart-red':    '#EF4444',
        'text-primary': '#0F1F33',  /** rich slate */
        'text-secondary':'#44556B', /** muted steel */
        'text-muted':   '#7C8BA1',  /** subtle labels */
      },
      fontFamily: {
        mono: ['IBM Plex Mono', 'SFMono-Regular', 'monospace'],
        sans: ['Manrope', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'glass': '0 1px 1px rgba(148, 163, 184, 0.2), 0 12px 24px -12px rgba(15, 31, 51, 0.22)',
        'float': '0 18px 36px -20px rgba(11, 58, 99, 0.4), 0 8px 18px -12px rgba(15, 31, 51, 0.3)',
      },
      borderRadius: {
        xl: '12px',
        '2xl': '16px',
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

