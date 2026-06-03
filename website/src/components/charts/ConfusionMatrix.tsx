'use client';

import { motion } from 'framer-motion';

interface ConfusionMatrixProps {
  matrix: number[][];
  labels: string[];
  normalized?: boolean;
}

export default function ConfusionMatrix({ matrix, labels, normalized = true }: ConfusionMatrixProps) {
  const maxVal = Math.max(...matrix.flat());

  const getColor = (value: number) => {
    const intensity = normalized ? value : value / Math.max(maxVal, 1);
    if (intensity > 0.7) return 'bg-indigo-500/80';
    if (intensity > 0.4) return 'bg-indigo-500/40';
    if (intensity > 0.2) return 'bg-indigo-500/20';
    return 'bg-white/5';
  };

  return (
    <div className="inline-block">
      {/* Header row */}
      <div className="flex items-end mb-2 ml-20">
        <span className="text-xs text-slate-400 mb-1 w-full text-center">Predicted</span>
      </div>
      <div className="flex">
        {/* Y-axis label */}
        <div className="flex items-center mr-2">
          <span className="text-xs text-slate-400 -rotate-90 whitespace-nowrap">Actual</span>
        </div>

        <div>
          {/* Column labels */}
          <div className="flex ml-16">
            {labels.map((label) => (
              <div key={`col-${label}`} className="w-20 text-center text-xs text-slate-400 mb-1">
                {label}
              </div>
            ))}
          </div>

          {/* Matrix cells */}
          {matrix.map((row, i) => (
            <div key={i} className="flex items-center mb-1">
              <span className="text-xs text-slate-400 w-14 text-right mr-2">{labels[i]}</span>
              {row.map((value, j) => (
                <motion.div
                  key={`${i}-${j}`}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: (i * labels.length + j) * 0.05 }}
                  className={`w-20 h-14 flex items-center justify-center rounded-lg mx-0.5 ${getColor(value)} border border-white/5`}
                >
                  <span className="text-sm font-mono text-white">
                    {normalized ? value.toFixed(3) : value}
                  </span>
                </motion.div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
