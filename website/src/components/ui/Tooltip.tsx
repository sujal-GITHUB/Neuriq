import * as React from "react"
import { cn } from "@/lib/utils"

export interface TooltipProps {
  content: React.ReactNode
  children: React.ReactNode
  className?: string
}

export function Tooltip({ content, children, className }: TooltipProps) {
  return (
    <div className="group relative inline-block">
      {children}
      <div className={cn(
        "absolute z-50 invisible opacity-0 group-hover:visible group-hover:opacity-100 transition-all duration-150 delay-300 pointer-events-none",
        "bottom-full left-1/2 -translate-x-1/2 mb-2",
        "bg-foreground text-background text-xs px-2 py-1 rounded-md max-w-[200px] w-max text-center leading-tight shadow-md",
        className
      )}>
        {content}
      </div>
    </div>
  )
}
