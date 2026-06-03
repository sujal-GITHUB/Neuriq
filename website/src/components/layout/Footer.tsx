import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="border-t border-border mt-24">
      <div className="max-w-7xl mx-auto px-6 py-8 flex items-center justify-between">
        <p className="text-xs text-foreground-subtle">
          Neuriq © {new Date().getFullYear()}
        </p>
        <div className="flex gap-4 text-xs text-foreground-subtle">
          <Link href="/docs" className="hover:text-foreground transition-colors">Documentation</Link>
          <Link href="/github" className="hover:text-foreground transition-colors">GitHub</Link>
          <Link href="/citations" className="hover:text-foreground transition-colors">Citations</Link>
        </div>
      </div>
    </footer>
  )
}
