import * as React from "react"
import { cn } from "@/lib/utils"

export interface SeparatorProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string
}

function Separator({ className, label, ...props }: SeparatorProps) {
  if (label) {
    return (
      <div className={cn("flex items-center gap-4 w-full", className)} {...props}>
        <div className="h-px w-full flex-1 bg-border" />
        <span className="text-xs text-foreground-subtle">{label}</span>
        <div className="h-px w-full flex-1 bg-border" />
      </div>
    )
  }

  return <div className={cn("h-px w-full bg-border", className)} {...props} />
}

export { Separator }
