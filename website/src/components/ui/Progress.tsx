import * as React from "react"
import { cn } from "@/lib/utils"

export interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number // 0 to 100
  indicatorClassName?: string
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value, indicatorClassName, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn("relative h-1.5 w-full overflow-hidden rounded-full bg-background-muted", className)}
        {...props}
      >
        <div
          className={cn("h-full w-full flex-1 bg-brand transition-all duration-[400ms] ease-out", indicatorClassName)}
          style={{ transform: `translateX(-${100 - (value || 0)}%)` }}
        />
      </div>
    )
  }
)
Progress.displayName = "Progress"

export { Progress }
