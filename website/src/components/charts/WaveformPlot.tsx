'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface WaveformPlotProps {
  signals: number[][];
  channels: string[];
  samplingRate: number;
  maxSeconds?: number;
  maxChannels?: number;
}

const CHANNEL_COLORS = [
  '#6366f1', '#06b6d4', '#22c55e', '#f59e0b', '#ef4444',
  '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#64748b'
];

export default function WaveformPlot({
  signals,
  channels,
  samplingRate,
  maxSeconds = 5,
  maxChannels = 5
}: WaveformPlotProps) {
  const displayChannels = channels.slice(0, maxChannels);
  const displaySignals = signals.slice(0, maxChannels);
  const maxSamples = samplingRate * maxSeconds;

  // Build data array
  const data = [];
  const sampleCount = Math.min(displaySignals[0]?.length || 0, maxSamples);

  // Downsample for performance (max 500 points)
  const step = Math.max(1, Math.floor(sampleCount / 500));

  for (let i = 0; i < sampleCount; i += step) {
    const point: Record<string, number> = {
      time: parseFloat((i / samplingRate).toFixed(3))
    };
    displayChannels.forEach((ch, chIdx) => {
      point[ch] = displaySignals[chIdx]?.[i] || 0;
    });
    data.push(point);
  }

  return (
    <div className="w-full h-[400px]">
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="time"
            tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#94a3b8', fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: '#64748b', fontSize: 10 }}
            label={{ value: 'Amplitude (µV)', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid rgba(99,102,241,0.3)',
              borderRadius: '8px',
              color: '#e2e8f0',
              fontSize: '12px'
            }}
          />
          <Legend />
          {displayChannels.map((ch, i) => (
            <Line
              key={ch}
              type="monotone"
              dataKey={ch}
              stroke={CHANNEL_COLORS[i % CHANNEL_COLORS.length]}
              strokeWidth={1.5}
              dot={false}
              name={ch}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
