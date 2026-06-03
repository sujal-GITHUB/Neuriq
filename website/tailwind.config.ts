import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        "background-subtle": "hsl(var(--background-subtle))",
        "background-muted": "hsl(var(--background-muted))",
        foreground: "hsl(var(--foreground))",
        "foreground-muted": "hsl(var(--foreground-muted))",
        "foreground-subtle": "hsl(var(--foreground-subtle))",
        border: "hsl(var(--border))",
        "border-strong": "hsl(var(--border-strong))",
        brand: "hsl(var(--brand))",
        "brand-foreground": "hsl(var(--brand-foreground))",
        "brand-muted": "hsl(var(--brand-muted))",
        surface: "hsl(var(--surface))",
        "surface-raised": "hsl(var(--surface-raised))",
        "level-low": "hsl(var(--level-low))",
        "level-low-bg": "hsl(var(--level-low-bg))",
        "level-moderate": "hsl(var(--level-moderate))",
        "level-moderate-bg": "hsl(var(--level-moderate-bg))",
        "level-high": "hsl(var(--level-high))",
        "level-high-bg": "hsl(var(--level-high-bg))",
        ring: "hsl(var(--ring))",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [],
};

export default config;
