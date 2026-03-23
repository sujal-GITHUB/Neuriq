"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/Button"
import { useTheme } from "next-themes"
import { Sun, Moon } from "lucide-react"

export default function Navbar() {
  const pathname = usePathname()
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  React.useEffect(() => setMounted(true), [])

  const navLinks = [
    { label: "Analysis", href: "/analyze" },
    { label: "Results", href: "/results" },
    { label: "Models", href: "/models" },
    { label: "Training", href: "/train" },
    { label: "Research", href: "/research" },
  ]

  return (
    <nav className="fixed top-0 left-0 w-full h-14 bg-background/80 backdrop-blur-md border-b border-border z-50">
      <div className="flex items-center justify-between max-w-7xl mx-auto px-6 h-full">
        <div className="flex items-center">
          <Link href="/" className="text-sm font-semibold text-foreground">Neuriq</Link>
          <div className="w-px h-4 bg-border mx-4" />
          <span className="text-xs text-foreground-subtle">v1.0</span>
        </div>

        <div className="hidden md:flex items-center space-x-6">
          {navLinks.map((link) => {
            const isActive = pathname === link.href || pathname.startsWith(link.href + '/')
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm font-medium transition-colors hover:text-foreground flex items-center h-14 ${
                  isActive ? "text-foreground border-b-2 border-brand" : "text-foreground-muted border-b-2 border-transparent"
                }`}
              >
                {link.label}
              </Link>
            )
          })}
        </div>

        <div className="flex items-center space-x-4">
          {mounted && (
            <Button
              variant="ghost"
              size="sm"
              className="w-8 h-8 p-0"
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              aria-label="Toggle theme"
            >
              <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            </Button>
          )}
          <div className="w-px h-4 bg-border hidden md:block" />
          <Link href="/analyze" className="hidden md:block">
            <Button size="sm">New Analysis</Button>
          </Link>
        </div>
      </div>
    </nav>
  )
}
