'use client';

import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer, Tooltip
} from 'recharts';
import type { BandPowers } from '@/types';

interface BandPowerRadarProps {
  bandPowers: BandPowers;
}

export default function BandPowerRadar({ bandPowers }: BandPowerRadarProps) {
  const data = [
    { band: 'Delta (0.5-4 Hz)', power: bandPowers.delta, fullMark: 50 },
    { band: 'Theta (4-8 Hz)', power: bandPowers.theta, fullMark: 50 },
    { band: 'Alpha (8-13 Hz)', power: bandPowers.alpha, fullMark: 50 },
    { band: 'Beta (13-30 Hz)', power: bandPowers.beta, fullMark: 50 },
    { band: 'Gamma (30-45 Hz)', power: bandPowers.gamma, fullMark: 50 },
  ];

  return (
    <div className="w-full h-[350px]">
      <ResponsiveContainer>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis
            dataKey="band"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
          />
          <PolarRadiusAxis
            angle={90}
            tick={{ fill: '#64748b', fontSize: 10 }}
            axisLine={false}
          />
          <Radar
            name="Power (µV²/Hz)"
            dataKey="power"
            stroke="#6366f1"
            fill="#6366f1"
            fillOpacity={0.25}
            strokeWidth={2}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid rgba(99,102,241,0.3)',
              borderRadius: '8px',
              color: '#e2e8f0'
            }}
            formatter={(value: number) => [`${value.toFixed(2)} µV²/Hz`, 'Power']}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
