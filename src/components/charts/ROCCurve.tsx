'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ROCCurveProps {
  curves: Record<string, { fpr: number[]; tpr: number[] }>;
}

const COLORS: Record<string, string> = {
  Low: '#22c55e',
  Moderate: '#f59e0b',
  High: '#ef4444',
};

export default function ROCCurve({ curves }: ROCCurveProps) {
  // Merge all curves into a single dataset indexed by FPR
  const allFpr = new Set<number>();
  Object.values(curves).forEach(c => c.fpr.forEach(f => allFpr.add(f)));
  const sortedFpr = Array.from(allFpr).sort((a, b) => a - b);

  const data = sortedFpr.map(fpr => {
    const point: Record<string, number> = { fpr };
    Object.entries(curves).forEach(([name, curve]) => {
      // Find closest TPR for this FPR
      let closestIdx = 0;
      let closestDist = Infinity;
      curve.fpr.forEach((f, i) => {
        const dist = Math.abs(f - fpr);
        if (dist < closestDist) {
          closestDist = dist;
          closestIdx = i;
        }
      });
      point[name] = curve.tpr[closestIdx];
    });
    return point;
  });

  return (
    <div className="w-full h-[400px]">
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="fpr" type="number" domain={[0, 1]}
            tick={{ fill: '#64748b', fontSize: 11 }}
            label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5, fill: '#94a3b8', fontSize: 12 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: '#64748b', fontSize: 11 }}
            label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fill: '#94a3b8', fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid rgba(99,102,241,0.3)',
              borderRadius: '8px',
              color: '#e2e8f0'
            }}
          />
          <Legend />
          {/* Diagonal reference line */}
          <Line
            dataKey="fpr" stroke="rgba(255,255,255,0.15)" strokeDasharray="5 5"
            dot={false} name="Random" legendType="none"
          />
          {Object.keys(curves).map((name) => (
            <Line
              key={name}
              dataKey={name}
              stroke={COLORS[name] || '#6366f1'}
              strokeWidth={2}
              dot={false}
              name={`${name} (ROC)`}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
