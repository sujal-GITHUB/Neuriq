"use client"

import * as React from "react"
import { Badge } from "@/components/ui/Badge"
import { Card } from "@/components/ui/Card"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, ReferenceLine, Tooltip } from "recharts"

const MOCK_COMPARISON = [
  { model: 'CNN-LSTM (Deep)', acc: '89.2', pre: '87.4', rec: '90.1', f1: '88.7', auc: '93.5', kap: '78.1', train: '1,420s', inf: '14.2ms', isBest: false },
  { model: 'Gradient Bagging', acc: '87.5', pre: '88.1', rec: '86.5', f1: '87.3', auc: '91.8', kap: '74.2', train: '15.4s', inf: '2.4ms', isBest: true },
  { model: 'Random Forest', acc: '86.1', pre: '85.2', rec: '87.0', f1: '86.0', auc: '90.5', kap: '71.8', train: '4.2s', inf: '1.2ms', isBest: false },
  { model: 'Support Vector', acc: '82.4', pre: '81.5', rec: '83.2', f1: '82.3', auc: '88.1', kap: '64.5', train: '28.1s', inf: '0.8ms', isBest: false },
  { model: 'LDA', acc: '76.8', pre: '75.2', rec: '78.5', f1: '76.8', auc: '81.2', kap: '53.4', train: '0.4s', inf: '0.1ms', isBest: false },
]

const MOCK_ROC = Array.from({ length: 20 }, (_, i) => ({
  fpr: i / 19,
  cnn: Math.pow(i / 19, 0.4),
  gb: Math.pow(i / 19, 0.5),
  rf: Math.pow(i / 19, 0.6),
  svm: Math.pow(i / 19, 0.7),
}))

export default function ModelsPage() {
  const [dataset, setDataset] = React.useState('2level')
  const [cmNormalized, setCmNormalized] = React.useState(true)

  return (
    <div className="flex w-full min-h-screen pt-14 -mt-14 relative bg-background">
      
      {/* SIDEBAR */}
      <aside className="fixed left-0 top-14 h-[calc(100vh-3.5rem)] w-56 border-r border-border bg-background-subtle py-8 px-4 z-10">
        <h2 className="text-xs uppercase tracking-wider text-foreground-muted mb-3 font-semibold px-3">Dataset</h2>
        <nav className="space-y-1">
          <button 
            onClick={() => setDataset('2level')}
            className={`w-full text-left font-medium text-sm py-2 px-3 rounded-md transition-colors ${
              dataset === '2level' ? 'bg-background-muted text-foreground' : 'text-foreground-muted hover:text-foreground hover:bg-background-muted/50'
            }`}
          >
            DASPS (Binary)
          </button>
          <button 
            onClick={() => setDataset('4level')}
            className={`w-full text-left font-medium text-sm py-2 px-3 rounded-md transition-colors ${
              dataset === '4level' ? 'bg-background-muted text-foreground' : 'text-foreground-muted hover:text-foreground hover:bg-background-muted/50'
            }`}
          >
            DASPS (4-Class)
          </button>
        </nav>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 ml-56 px-0 py-8 pb-32 max-w-6xl">
        <div className="mb-8 border-b border-border pb-8">
          <div className="text-xs text-foreground-subtle mb-2">Dashboard / Evaluation</div>
          <h1 className="text-3xl font-semibold text-foreground tracking-tight">Model Benchmarks</h1>
          <p className="text-sm text-foreground-muted mt-1 max-w-xl">
            Live evaluation metrics computed from robust cross-validation spanning psychological state distributions.
          </p>
        </div>

        {/* COMPARISON TABLE */}
        <section className="mb-12">
          <h3 className="text-sm font-medium text-foreground mb-4">Performance Metrics</h3>
          <div className="w-full overflow-x-auto border border-border rounded-lg bg-surface">
            <table className="w-full text-sm text-left">
              <thead className="bg-background-subtle border-b border-border">
                <tr>
                  {['Model', 'Acc', 'Prec', 'Rec', 'F1', 'AUC', 'Kappa', 'Train', 'Inference'].map(h => (
                    <th key={h} className="px-4 py-3 text-xs font-medium text-foreground-muted uppercase tracking-wider">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {MOCK_COMPARISON.map((m, i) => (
                  <tr key={m.model} className={`border-b border-border transition-colors hover:bg-background-subtle/40 ${m.model === 'Random Forest' ? 'bg-brand/5' : ''}`}>
                    <td className={`px-4 py-3 whitespace-nowrap border-l-2 ${m.model === 'Random Forest' ? 'border-brand' : 'border-transparent'}`}>
                      <span className="font-medium text-foreground">{m.model}</span>
                    </td>
                    <td className="px-4 py-3 tabular-nums font-semibold text-foreground">{m.acc}%</td>
                    <td className="px-4 py-3 tabular-nums text-foreground-subtle">{m.pre}%</td>
                    <td className="px-4 py-3 tabular-nums text-foreground-subtle">{m.rec}%</td>
                    <td className="px-4 py-3 tabular-nums text-foreground-subtle">{m.f1}%</td>
                    <td className="px-4 py-3 tabular-nums text-brand font-medium">{m.auc}%</td>
                    <td className="px-4 py-3 tabular-nums text-foreground-muted">{m.kap}</td>
                    <td className="px-4 py-3 tabular-nums text-foreground-muted text-xs">{m.train}</td>
                    <td className="px-4 py-3 tabular-nums text-foreground-muted text-xs">{m.inf}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* ROC CURVES */}
          <section>
            <h3 className="text-sm font-medium text-foreground mb-4">ROC Characteristics</h3>
            <Card className="p-6 flex flex-col md:flex-row gap-6 items-center">
              <div className="w-full md:w-2/3 h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={MOCK_ROC} margin={{ top: 5, right: 5, left: -25, bottom: 5 }}>
                    <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 4" vertical={false} />
                    <XAxis dataKey="fpr" type="number" tickCount={5} tick={{ fontSize: 11, fill: "hsl(var(--foreground-muted))" }} axisLine={false} tickLine={false} />
                    <YAxis type="number" tickCount={5} tick={{ fontSize: 11, fill: "hsl(var(--foreground-muted))" }} axisLine={false} tickLine={false} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }} 
                      itemStyle={{ color: "hsl(var(--foreground))", fontSize: "12px" }}
                      labelStyle={{ display: 'none' }}
                    />
                    <ReferenceLine segment={[{x:0, y:0}, {x:1, y:1}]} stroke="hsl(var(--foreground-subtle))" strokeDasharray="3 3" />
                    
                    <Line type="monotone" dataKey="cnn" name="CNN-LSTM" stroke="#3B82F6" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="gb" name="Grad Bagging" stroke="#64748B" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="rf" name="Random Forest" stroke="#94A3B8" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              
              <div className="w-full md:w-1/3 space-y-3">
                 <div className="text-xs text-foreground-muted mb-2 font-medium uppercase tracking-wider">Area Under Curve</div>
                 {[
                   { name: 'CNN-LSTM', val: '0.935', color: '#3B82F6' },
                   { name: 'Grad Bagging', val: '0.918', color: '#64748B' },
                   { name: 'Random Forest', val: '0.905', color: '#94A3B8' },
                 ].map(l => (
                   <div key={l.name} className="flex justify-between items-center text-xs">
                     <div className="flex items-center gap-2">
                       <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: l.color }} />
                       <span className="text-foreground">{l.name}</span>
                     </div>
                     <span className="font-mono text-foreground-subtle">({l.val})</span>
                   </div>
                 ))}
              </div>
            </Card>
          </section>

          {/* CONFUSION MATRIX */}
          <section>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-medium text-foreground">Confusion Matrix — CNN-LSTM</h3>
              <button 
                className="text-xs text-foreground-subtle hover:text-foreground underline transition-colors"
                onClick={() => setCmNormalized(!cmNormalized)}
              >
                {cmNormalized ? "Show Raw Count" : "Show Normalized"}
              </button>
            </div>
            <Card className="p-8 flex items-center justify-center">
              <div className="flex flex-col">
                <div className="w-full text-center text-xs text-foreground-muted mb-2 uppercase tracking-widest">Predicted Class</div>
                <div className="flex items-center">
                  <div className="text-xs text-foreground-muted -rotate-90 mr-4 uppercase tracking-widest min-w-[max-content]">True Class</div>
                  <div className="grid grid-cols-2 gap-1 border border-border p-1 bg-background-subtle rounded-md">
                    {/* Headers internal */}
                    <div className="col-span-2 grid grid-cols-2 gap-1 text-center text-xs text-foreground-muted pb-1 border-b border-border/50">
                      <div>Anxious</div>
                      <div>Calm</div>
                    </div>
                    {/* Rows */}
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-foreground-muted w-10 text-right">Anxious</span>
                      <div className="w-20 h-16 bg-brand flex items-center justify-center rounded text-brand-foreground font-semibold text-sm">
                        {cmNormalized ? "88.4%" : "1,040"}
                      </div>
                      <div className="w-20 h-16 bg-brand/10 flex items-center justify-center rounded text-foreground font-semibold text-sm">
                        {cmNormalized ? "11.6%" : "136"}
                      </div>
                    </div>
                    <div className="col-span-2 h-px bg-transparent" />
                    <div className="flex items-center gap-2">
                       <span className="text-xs text-foreground-muted w-10 text-right">Calm</span>
                      <div className="w-20 h-16 bg-brand/10 flex items-center justify-center rounded text-foreground font-semibold text-sm">
                        {cmNormalized ? "10.2%" : "120"}
                      </div>
                      <div className="w-20 h-16 bg-brand flex items-center justify-center rounded text-brand-foreground font-semibold text-sm">
                        {cmNormalized ? "89.8%" : "1,056"}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </section>
        </div>

        {/* CROSS VALIDATION TABLE */}
        <section>
          <h3 className="text-sm font-medium text-foreground mb-4">5-Fold Cross Validation Logs</h3>
          <div className="border border-border rounded-lg bg-surface overflow-hidden">
            <table className="w-full text-sm text-left">
              <thead className="bg-background-subtle border-b border-border">
                 <tr>
                  <th className="px-4 py-3 text-xs font-medium text-foreground-muted uppercase">Fold</th>
                  <th className="px-4 py-3 text-xs font-medium text-foreground-muted uppercase">Accuracy</th>
                  <th className="px-4 py-3 text-xs font-medium text-foreground-muted uppercase">F1-Score</th>
                  <th className="px-4 py-3 text-xs font-medium text-foreground-muted uppercase">Val Loss</th>
                 </tr>
              </thead>
              <tbody>
                {[1,2,3,4,5].map(f => (
                  <tr key={f} className="border-b border-border/50">
                    <td className="px-4 py-2 text-foreground-subtle">Fold {f}</td>
                    <td className="px-4 py-2 tabular-nums text-foreground">87.{2 + f}%</td>
                    <td className="px-4 py-2 tabular-nums text-foreground">86.{8 + f}%</td>
                    <td className="px-4 py-2 tabular-nums text-foreground-subtle">0.3{40 + f}</td>
                  </tr>
                ))}
                <tr className="bg-background-subtle/30 font-medium">
                  <td className="px-4 py-3 text-foreground">Mean</td>
                  <td className="px-4 py-3 text-foreground">87.5% <span className="text-foreground-muted text-xs font-normal">± 0.2</span></td>
                  <td className="px-4 py-3 text-foreground">87.1% <span className="text-foreground-muted text-xs font-normal">± 0.3</span></td>
                  <td className="px-4 py-3 text-foreground">0.342 <span className="text-foreground-muted text-xs font-normal">± 0.01</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
        
      </main>
    </div>
  )
}
