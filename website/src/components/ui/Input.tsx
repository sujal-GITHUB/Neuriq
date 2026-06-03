import * as React from "react"
import { cn } from "@/lib/utils"

export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'prefix'> {
  label: string
  helperText?: string
  error?: string
  unit?: string
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, label, helperText, error, unit, ...props }, ref) => {
    return (
      <div className="w-full flex flex-col gap-1">
        <label className="text-xs font-medium text-foreground-muted">{label}</label>
        <div className="relative flex items-center">
          <input
            type={type}
            className={cn(
              "flex h-9 w-full rounded-md border border-border bg-background px-3 py-1 text-sm transition-colors duration-150 ease-out file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-foreground-subtle focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand focus-visible:border-brand disabled:cursor-not-allowed disabled:opacity-50",
              error && "border-level-high focus-visible:ring-level-high focus-visible:border-level-high",
              unit && "pr-8",
              className
            )}
            ref={ref}
            {...props}
          />
          {unit && (
            <span className="absolute right-3 text-xs text-foreground-subtle pointer-events-none">
              {unit}
            </span>
          )}
        </div>
        {(helperText || error) && (
          <span className={cn("text-xs", error ? "text-level-high" : "text-foreground-muted")}>
            {error || helperText}
          </span>
        )}
      </div>
    )
  }
)
Input.displayName = "Input"

export { Input }
