'use client';

import { motion } from 'framer-motion';

interface ConfidenceBarProps {
  confidence: number;
  probabilities: { Low: number; Moderate: number; High: number };
}

export default function ConfidenceBar({ confidence, probabilities }: ConfidenceBarProps) {
  return (
    <div className="space-y-4">
      {/* Main confidence */}
      <div>
        <div className="flex justify-between mb-1.5">
          <span className="text-sm text-slate-400">Confidence</span>
          <span className="text-sm font-mono text-white">{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="h-3 bg-white/5 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 1, ease: 'easeOut' }}
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500"
          />
        </div>
      </div>

      {/* Per-class probabilities */}
      <div className="space-y-3">
        {[
          { label: 'Low Anxiety', value: probabilities.Low, color: '#22c55e' },
          { label: 'Moderate Anxiety', value: probabilities.Moderate, color: '#f59e0b' },
          { label: 'High Anxiety', value: probabilities.High, color: '#ef4444' },
        ].map((item) => (
          <div key={item.label}>
            <div className="flex justify-between mb-1">
              <span className="text-xs text-slate-400">{item.label}</span>
              <span className="text-xs font-mono text-slate-300">{(item.value * 100).toFixed(1)}%</span>
            </div>
            <div className="h-2 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${item.value * 100}%` }}
                transition={{ duration: 0.8, ease: 'easeOut', delay: 0.2 }}
                className="h-full rounded-full"
                style={{ backgroundColor: item.color }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
