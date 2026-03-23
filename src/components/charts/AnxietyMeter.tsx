'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

interface AnxietyMeterProps {
  level: 'Low' | 'Moderate' | 'High';
  confidence: number;
}

export default function AnxietyMeter({ level, confidence }: AnxietyMeterProps) {
  const [animatedAngle, setAnimatedAngle] = useState(0);

  const targetAngle = level === 'Low' ? 30 : level === 'Moderate' ? 90 : 150;
  const color = level === 'Low' ? '#22c55e' : level === 'Moderate' ? '#f59e0b' : '#ef4444';

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedAngle(targetAngle), 300);
    return () => clearTimeout(timer);
  }, [targetAngle]);

  const radius = 120;
  const cx = 150;
  const cy = 140;

  const needleX = cx + radius * 0.85 * Math.cos((Math.PI * (180 - animatedAngle)) / 180);
  const needleY = cy - radius * 0.85 * Math.sin((Math.PI * (180 - animatedAngle)) / 180);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 300 180" className="w-full max-w-[400px]">
        <defs>
          <linearGradient id="gaugeGradient" x1="0" x2="1">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Background arc */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="24"
          strokeLinecap="round"
        />

        {/* Colored arc */}
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth="24"
          strokeLinecap="round"
          opacity="0.8"
        />

        {/* Tick marks */}
        {[0, 30, 60, 90, 120, 150, 180].map((angle) => {
          const x1 = cx + (radius - 18) * Math.cos((Math.PI * (180 - angle)) / 180);
          const y1 = cy - (radius - 18) * Math.sin((Math.PI * (180 - angle)) / 180);
          const x2 = cx + (radius + 18) * Math.cos((Math.PI * (180 - angle)) / 180);
          const y2 = cy - (radius + 18) * Math.sin((Math.PI * (180 - angle)) / 180);
          return (
            <line key={angle} x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="rgba(255,255,255,0.3)" strokeWidth="2" />
          );
        })}

        {/* Labels */}
        <text x="30" y={cy + 25} fill="#22c55e" fontSize="11" fontWeight="600">Low</text>
        <text x={cx - 18} y="30" fill="#f59e0b" fontSize="11" fontWeight="600">Moderate</text>
        <text x="235" y={cy + 25} fill="#ef4444" fontSize="11" fontWeight="600">High</text>

        {/* Needle */}
        <motion.line
          x1={cx}
          y1={cy}
          animate={{ x2: needleX, y2: needleY }}
          transition={{ type: 'spring', stiffness: 60, damping: 15 }}
          stroke={color}
          strokeWidth="3"
          strokeLinecap="round"
          filter="url(#glow)"
        />

        {/* Center dot */}
        <circle cx={cx} cy={cy} r="8" fill={color} opacity="0.9" />
        <circle cx={cx} cy={cy} r="4" fill="white" opacity="0.9" />
      </svg>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="text-center -mt-2"
      >
        <span
          className="text-3xl font-bold"
          style={{ color }}
        >
          {level} Anxiety
        </span>
        <p className="text-sm text-slate-400 mt-1">
          {(confidence * 100).toFixed(1)}% confidence
        </p>
      </motion.div>
    </div>
  );
}
