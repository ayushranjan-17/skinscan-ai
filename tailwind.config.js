/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        derma: {
          bg: '#070B14',
          card: '#0D1527',
          cardHover: '#131E38',
          surface: '#17223F',
          border: '#1F2E54',
          borderLight: '#2D4075',
          accent: '#00F0FF',
          accentHover: '#00D0DF',
          teal: '#00D084',
          blue: '#0693E3',
          purple: '#9B51E0',
          yellow: '#FFDC4B',
          amber: '#F59E0B',
          coral: '#EF4444',
          rose: '#F43F5E',
          muted: '#8B9BB4',
        },
      },
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 25px -5px rgba(0, 240, 255, 0.25)',
        'glow-teal': '0 0 25px -5px rgba(0, 208, 132, 0.25)',
        'glow-coral': '0 0 25px -5px rgba(239, 68, 68, 0.3)',
        'glow-amber': '0 0 25px -5px rgba(245, 158, 11, 0.25)',
        glass: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 12s linear infinite',
      },
    },
  },
  plugins: [],
};
