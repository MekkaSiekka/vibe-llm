/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: ["class"],
  theme: {
    extend: {
      colors: {
        // iOS 18 inspired design system
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: "hsl(var(--card))",
        "card-foreground": "hsl(var(--card-foreground))",
        popover: "hsl(var(--popover))",
        "popover-foreground": "hsl(var(--popover-foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          "1": "hsl(var(--chart-1))",
          "2": "hsl(var(--chart-2))",
          "3": "hsl(var(--chart-3))",
          "4": "hsl(var(--chart-4))",
          "5": "hsl(var(--chart-5))",
        },
        // iOS specific colors
        "ios-blue": "#007AFF",
        "ios-purple": "#5856D6", 
        "ios-green": "#34C759",
        "ios-orange": "#FF9500",
        "ios-red": "#FF3B30",
        "ios-yellow": "#FFCC00",
        "ios-gray": {
          50: "#F2F2F7",
          100: "#E5E5EA", 
          200: "#D1D1D6",
          300: "#C7C7CC",
          400: "#AEAEB2",
          500: "#8E8E93",
          600: "#636366",
          700: "#48484A",
          800: "#3A3A3C",
          900: "#2C2C2E",
          950: "#1C1C1E",
        }
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        // iOS specific radius
        "ios-sm": "8px",
        "ios-md": "12px", 
        "ios-lg": "16px",
        "ios-xl": "20px",
        "ios-2xl": "24px",
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont", 
          "SF Pro Display",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "SF Mono",
          "Monaco",
          "Inconsolata",
          "Roboto Mono",
          "Consolas",
          "Courier New",
          "monospace",
        ],
      },
      fontSize: {
        // iOS typography scale
        "ios-xs": ["12px", { lineHeight: "16px" }],
        "ios-sm": ["14px", { lineHeight: "20px" }],
        "ios-base": ["16px", { lineHeight: "24px" }],
        "ios-lg": ["18px", { lineHeight: "28px" }],
        "ios-xl": ["20px", { lineHeight: "28px" }],
        "ios-2xl": ["24px", { lineHeight: "32px" }],
        "ios-3xl": ["28px", { lineHeight: "36px" }],
        "ios-4xl": ["32px", { lineHeight: "40px" }],
        "ios-5xl": ["36px", { lineHeight: "44px" }],
      },
      spacing: {
        // iOS spacing system
        "ios-xs": "4px",
        "ios-sm": "8px", 
        "ios-md": "16px",
        "ios-lg": "24px",
        "ios-xl": "32px",
        "ios-2xl": "48px",
        "ios-3xl": "64px",
      },
      boxShadow: {
        // iOS shadows
        "ios-sm": "0 1px 3px rgba(0, 0, 0, 0.08)",
        "ios-md": "0 4px 12px rgba(0, 0, 0, 0.1)", 
        "ios-lg": "0 8px 24px rgba(0, 0, 0, 0.12)",
        "ios-xl": "0 16px 40px rgba(0, 0, 0, 0.15)",
      },
      animation: {
        "fade-in": "fadeIn 0.3s ease-out",
        "slide-up": "slideUp 0.3s ease-out", 
        "slide-down": "slideDown 0.3s ease-out",
        "scale-in": "scaleIn 0.2s ease-out",
        "bounce-subtle": "bounceSubtle 0.6s ease-out",
        "pulse-subtle": "pulseSubtle 2s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideDown: {
          "0%": { opacity: "0", transform: "translateY(-10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        scaleIn: {
          "0%": { opacity: "0", transform: "scale(0.95)" },
          "100%": { opacity: "1", transform: "scale(1)" },
        },
        bounceSubtle: {
          "0%, 20%, 53%, 80%, 100%": { transform: "translate3d(0,0,0)" },
          "40%, 43%": { transform: "translate3d(0,-8px,0)" },
          "70%": { transform: "translate3d(0,-4px,0)" },
          "90%": { transform: "translate3d(0,-2px,0)" },
        },
        pulseSubtle: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.8" },
        },
        shimmer: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
      },
      backdropBlur: {
        "ios": "20px",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
}
