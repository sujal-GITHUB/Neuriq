import * as React from "react"
import { cn } from "@/lib/utils"
import { ChevronUp, ChevronDown } from "lucide-react"

export interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (row: T) => React.ReactNode;
  sortable?: boolean;
}

export interface DataTableProps<T> extends React.HTMLAttributes<HTMLDivElement> {
  data: T[];
  columns: Column<T>[];
  striped?: boolean;
}

export function DataTable<T>({ data, columns, striped = false, className, ...props }: DataTableProps<T>) {
  const [sortConfig, setSortConfig] = React.useState<{ key: string, direction: 'asc'|'desc' } | null>(null);

  const requestSort = (key: string) => {
    let direction: 'asc'|'desc' = 'asc';
    if (sortConfig && sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const sortedData = React.useMemo(() => {
    if (!sortConfig) return data;
    return [...data].sort((a: any, b: any) => {
      const aVal = a[sortConfig.key];
      const bVal = b[sortConfig.key];
      if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
      if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }, [data, sortConfig]);

  return (
    <div className={cn("w-full overflow-x-auto border border-border rounded-lg", className)} {...props}>
      <table className="w-full text-sm text-left border-collapse">
        <thead className="bg-background-subtle">
          <tr className="border-b border-border">
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={cn(
                  "px-4 py-3 text-xs font-medium text-foreground-muted uppercase tracking-wider",
                  col.sortable && "cursor-pointer hover:text-foreground transition-colors"
                )}
                onClick={() => col.sortable && requestSort(String(col.key))}
              >
                <div className="flex items-center gap-1.5">
                  {col.header}
                  {col.sortable && (
                    <span className="flex flex-col text-foreground-subtle">
                      <ChevronUp className={cn("h-3 w-3 -mb-1", sortConfig?.key === col.key && sortConfig.direction === 'asc' ? 'text-foreground' : 'opacity-50')} />
                      <ChevronDown className={cn("h-3 w-3", sortConfig?.key === col.key && sortConfig.direction === 'desc' ? 'text-foreground' : 'opacity-50')} />
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.map((row, i) => (
            <tr
              key={i}
              className={cn(
                "border-b border-border hover:bg-background-subtle transition-colors",
                striped && i % 2 === 1 && "bg-background-subtle/50"
              )}
            >
              {columns.map((col) => (
                <td key={String(col.key)} className="px-4 py-3 text-foreground whitespace-nowrap">
                  {col.render ? col.render(row) : row[col.key as keyof T] as React.ReactNode}
                </td>
              ))}
            </tr>
          ))}
          {sortedData.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="px-4 py-8 text-center text-foreground-muted text-xs">
                No data available.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
