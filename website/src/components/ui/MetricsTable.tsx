'use client';

import type { FeatureImportance } from '@/types';

interface MetricsTableProps {
  features: FeatureImportance[];
}

export default function MetricsTable({ features }: MetricsTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-white/10">
            <th className="text-left py-3 px-4 text-xs font-medium text-slate-400 uppercase tracking-wider">#</th>
            <th className="text-left py-3 px-4 text-xs font-medium text-slate-400 uppercase tracking-wider">Feature</th>
            <th className="text-left py-3 px-4 text-xs font-medium text-slate-400 uppercase tracking-wider">Channel</th>
            <th className="text-left py-3 px-4 text-xs font-medium text-slate-400 uppercase tracking-wider">Band</th>
            <th className="text-right py-3 px-4 text-xs font-medium text-slate-400 uppercase tracking-wider">SHAP Value</th>
            <th className="text-center py-3 px-4 text-xs font-medium text-slate-400 uppercase tracking-wider">Direction</th>
          </tr>
        </thead>
        <tbody>
          {features.map((f, i) => (
            <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
              <td className="py-3 px-4 text-slate-500 font-mono text-xs">{i + 1}</td>
              <td className="py-3 px-4 text-white font-medium text-xs">{f.name}</td>
              <td className="py-3 px-4">
                <span className="px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-400 text-xs border border-indigo-500/20">
                  {f.channel || 'N/A'}
                </span>
              </td>
              <td className="py-3 px-4 text-slate-400 text-xs">{f.band || 'N/A'}</td>
              <td className="py-3 px-4 text-right font-mono text-xs text-slate-300">
                {(f.importance || f.mean_shap || 0).toFixed(4)}
              </td>
              <td className="py-3 px-4 text-center">
                {f.direction === 'positive' ? (
                  <span className="text-red-400 text-xs">↑ Anxiety</span>
                ) : (
                  <span className="text-green-400 text-xs">↓ Relax</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
