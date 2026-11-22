/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Original navy surfaces are kept as-is so exports match earlier printed reports.
        // Accent and status colors are the same hues, with the neon taken down.
        derma: {
          bg: '#070B14',
          card: '#0D1527',
          cardHover: '#131E38',
          surface: '#17223F',
          border: '#1F2E54',
          borderLight: '#2D4075',
          accent: '#4FD1E8',
          accentHover: '#74DCEE',
          teal: '#34C38F',
          blue: '#4A9FE0',
          purple: '#A77FDB',
          yellow: '#EBC95A',
          amber: '#E8A33D',
          coral: '#E5645A',
          rose: '#EB6B7E',
          muted: '#8B9BB4',
        },
      },
      fontFamily: {
        sans: ['Figtree', 'system-ui', 'sans-serif'],
        display: ['Fraunces', 'Georgia', 'serif'],
        // Numbers and IDs use the body face with tabular figures instead of a code font.
        mono: ['Figtree', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        // Former neon glows are now plain soft shadows.
        'glow-cyan': '0 1px 2px rgba(0,0,0,0.25), 0 10px 24px -14px rgba(0,0,0,0.55)',
        'glow-teal': '0 1px 2px rgba(0,0,0,0.25), 0 10px 24px -14px rgba(0,0,0,0.55)',
        'glow-coral': '0 1px 2px rgba(0,0,0,0.25), 0 10px 24px -14px rgba(0,0,0,0.55)',
        'glow-amber': '0 1px 2px rgba(0,0,0,0.25), 0 10px 24px -14px rgba(0,0,0,0.55)',
        glass: '0 1px 2px rgba(0,0,0,0.25), 0 10px 24px -14px rgba(0,0,0,0.55)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 12s linear infinite',
      },
    },
  },
  plugins: [],
};
