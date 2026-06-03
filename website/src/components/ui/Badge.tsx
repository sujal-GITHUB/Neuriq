import * as React from "react"
import { cn } from "@/lib/utils"

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "low" | "moderate" | "high" | "outline"
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  const variants = {
    default: "bg-background-muted text-foreground",
    low: "bg-level-low-bg text-level-low",
    moderate: "bg-level-moderate-bg text-level-moderate",
    high: "bg-level-high-bg text-level-high",
    outline: "border border-border text-foreground-muted"
  }

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium transition-colors duration-150 ease-out",
        variants[variant],
        className
      )}
      {...props}
    />
  )
}

export { Badge }
