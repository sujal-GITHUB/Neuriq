'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface TrainingLossChartProps {
  history: Array<{
    epoch: number;
    train_loss: number;
    val_loss: number;
    train_accuracy: number;
    val_accuracy: number;
  }>;
}

export default function TrainingLossChart({ history }: TrainingLossChartProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Loss chart */}
      <div className="h-[300px]">
        <h4 className="text-sm font-medium text-slate-300 mb-2">Loss</h4>
        <ResponsiveContainer>
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid rgba(99,102,241,0.3)',
                borderRadius: '8px',
                color: '#e2e8f0'
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#6366f1" strokeWidth={2} dot={false} name="Train Loss" />
            <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" strokeWidth={2} dot={false} name="Val Loss" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Accuracy chart */}
      <div className="h-[300px]">
        <h4 className="text-sm font-medium text-slate-300 mb-2">Accuracy</h4>
        <ResponsiveContainer>
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="epoch" tick={{ fill: '#64748b', fontSize: 10 }} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 1]} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid rgba(99,102,241,0.3)',
                borderRadius: '8px',
                color: '#e2e8f0'
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="train_accuracy" stroke="#22c55e" strokeWidth={2} dot={false} name="Train Acc" />
            <Line type="monotone" dataKey="val_accuracy" stroke="#06b6d4" strokeWidth={2} dot={false} name="Val Acc" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
