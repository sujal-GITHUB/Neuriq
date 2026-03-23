import * as React from "react"
import { cn } from "@/lib/utils"
import { Loader2 } from "lucide-react"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "ghost" | "outline" | "destructive"
  size?: "sm" | "md" | "lg"
  loading?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "md", loading = false, children, disabled, ...props }, ref) => {
    const variants = {
      default: "bg-foreground text-background hover:bg-foreground/90",
      ghost: "bg-transparent hover:bg-background-muted text-foreground",
      outline: "border border-border bg-transparent hover:bg-background-subtle text-foreground",
      destructive: "bg-level-high text-level-high-bg hover:bg-level-high/90"
    }
    
    const sizes = {
      sm: "h-8 px-3 text-xs",
      md: "h-9 px-4 text-sm",
      lg: "h-10 px-6 text-sm"
    }

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={cn(
          "inline-flex items-center justify-center rounded-md font-medium transition-colors duration-150 ease-out",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          "disabled:opacity-40 disabled:cursor-not-allowed",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {loading ? (
          <>
            <Loader2 className="absolute h-4 w-4 animate-spin" />
            <span className="invisible">{children}</span>
          </>
        ) : (
          children
        )}
      </button>
    )
  }
)
Button.displayName = "Button"

export { Button }
