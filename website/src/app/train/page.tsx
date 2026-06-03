"use client"

import * as React from "react"
import { Badge } from "@/components/ui/Badge"
import { Card } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip } from "recharts"

const MOCK_HISTORY = Array.from({ length: 15 }, (_, i) => ({
  epoch: i + 1,
  trainLoss: Math.exp(-i / 5) + 0.1 + Math.random() * 0.05,
  valLoss: Math.exp(-i / 6) + 0.15 + Math.random() * 0.08,
  acc: (1 - Math.exp(-i / 4)) * 80 + 10 + Math.random() * 2,
}))

export default function TrainPage() {
  const [selectedDataset, setSelectedDataset] = React.useState('2level')
  const [selectedModel, setSelectedModel] = React.useState('gradient')
  const [useSmote, setUseSmote] = React.useState(true)
  const [evalType, setEvalType] = React.useState('kfold')
  const [isTraining, setIsTraining] = React.useState(false)

  const models = [
    { id: "random_forest", name: "Random Forest", desc: "Baseline decision trees" },
    { id: "gradient", name: "Gradient Bagging", desc: "Top performing ensemble" },
    { id: "svm", name: "Support Vector", desc: "Linear separation margin" },
    { id: "cnn", name: "CNN-LSTM", desc: "Deep spatio-temporal" }
  ]

  return (
    <div className="flex w-full min-h-[calc(100vh-3.5rem)] pt-14 -mt-14 relative bg-background">
      <main className="flex-1 py-8 px-0 max-w-7xl mx-auto">
        <div className="mb-8 border-b border-border pb-8">
          <div className="text-xs text-foreground-subtle mb-2">Dashboard / Optimization</div>
          <h1 className="text-3xl font-semibold text-foreground tracking-tight">Model Training Studio</h1>
          <p className="text-sm text-foreground-muted mt-1 max-w-xl">
            Configure architectures and hyperparameter grids for live continuous evaluation loops.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 pb-32">
          
          {/* CONFIGURATION PANEL */}
          <div className="col-span-1 space-y-8">
            
            <section>
              <h2 className="text-xs uppercase tracking-wider text-foreground-muted font-semibold mb-3">Dataset Schema</h2>
              <div className="space-y-2">
                <div 
                  onClick={() => setSelectedDataset('2level')}
                  className={`border rounded-lg p-3 cursor-pointer transition-colors ${selectedDataset === '2level' ? 'border-brand bg-brand/5' : 'border-border bg-surface hover:border-border-strong'}`}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-sm font-medium text-foreground">DASPS (Binary)</span>
                    <Badge variant={selectedDataset === '2level' ? 'default' : 'outline'}>23 Subj</Badge>
                  </div>
                  <p className="text-xs text-foreground-muted mt-1">414 trials • 14ch EEG</p>
                </div>
                <div 
                  onClick={() => setSelectedDataset('4level')}
                  className={`border rounded-lg p-3 cursor-pointer transition-colors ${selectedDataset === '4level' ? 'border-brand bg-brand/5' : 'border-border bg-surface hover:border-border-strong'}`}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-sm font-medium text-foreground">DASPS (4-Class)</span>
                    <Badge variant={selectedDataset === '4level' ? 'default' : 'outline'}>23 Subj</Badge>
                  </div>
                  <p className="text-xs text-foreground-muted mt-1">Detailed scale mapping • 14ch EEG</p>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-xs uppercase tracking-wider text-foreground-muted font-semibold mb-3">Architecture Selection</h2>
              <div className="grid grid-cols-1 gap-2">
                {models.map(m => (
                  <div 
                    key={m.id}
                    onClick={() => setSelectedModel(m.id)}
                    className={`border rounded-lg p-3 cursor-pointer transition-colors ${selectedModel === m.id ? 'border-brand bg-brand/5' : 'border-border bg-surface hover:border-border-strong'}`}
                  >
                    <span className="text-sm font-medium text-foreground">{m.name}</span>
                    <p className="text-xs text-foreground-muted mt-0.5">{m.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-xs uppercase tracking-wider text-foreground-muted font-semibold mb-3">Hyperparameters</h2>
              <div className="grid grid-cols-2 gap-4">
                <Input label="Estimators" defaultValue="100" />
                <Input label="Max Depth" defaultValue="6" />
                <Input label="Folds" defaultValue="5" />
                <div className="flex flex-col gap-1 justify-center">
                   <label className="text-xs font-medium text-foreground-muted mb-2">Use SMOTE</label>
                   <button 
                     onClick={() => setUseSmote(!useSmote)}
                     className={`w-12 h-6 rounded-full transition-colors relative flex items-center px-1 ${useSmote ? 'bg-brand' : 'bg-background-muted'}`}
                   >
                     <div className={`w-4 h-4 rounded-full bg-white transition-transform ${useSmote ? 'translate-x-6' : 'translate-x-0'}`} />
                   </button>
                </div>
              </div>
            </section>

            <section>
              <h2 className="text-xs uppercase tracking-wider text-foreground-muted font-semibold mb-3">Evaluation Method</h2>
              <div className="flex flex-col gap-2">
                 <div 
                   onClick={() => setEvalType('kfold')}
                   className={`border rounded-lg p-3 cursor-pointer transition-colors ${evalType === 'kfold' ? 'border-brand bg-brand/5' : 'border-border bg-surface hover:border-border-strong'}`}
                 >
                   <span className="text-sm font-medium text-foreground">K-Fold CV</span>
                 </div>
                 <div 
                   onClick={() => setEvalType('loso')}
                   className={`border rounded-lg p-3 cursor-pointer transition-colors ${evalType === 'loso' ? 'border-brand bg-brand/5' : 'border-border bg-surface hover:border-border-strong'}`}
                 >
                   <span className="text-sm font-medium text-foreground">Leave-One-Subject-Out (LOSO)</span>
                 </div>
              </div>
            </section>

            <div className="pt-2">
              <Button size="lg" className="w-full" onClick={() => setIsTraining(!isTraining)}>
                {isTraining ? 'Stop Evaluation' : 'Start Processing'}
              </Button>
            </div>
          </div>

          {/* TRAINING MONITOR */}
          <div className="col-span-2 space-y-6">
            
            <div className="flex justify-between items-center pb-4 border-b border-border">
              <div className="flex items-center gap-3">
                <div className="text-sm font-medium text-foreground">Compute Monitor</div>
                {isTraining ? (
                   <Badge className="bg-brand/20 text-brand">Running</Badge>
                ) : (
                   <Badge variant="outline">Idle Target</Badge>
                )}
              </div>
              <div className="text-sm tabular-nums text-foreground-muted font-mono">00:14:42 elapsed</div>
            </div>

            <Card className="p-6">
               <h3 className="text-sm font-medium text-foreground mb-4">Live Convergence Trace</h3>
               <div className="h-[280px] w-full">
                 <ResponsiveContainer width="100%" height="100%">
                   <LineChart data={isTraining ? MOCK_HISTORY : []} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                     <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="4 4" vertical={false} />
                     <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: "hsl(var(--foreground-muted))" }} axisLine={false} tickLine={false} />
                     <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "hsl(var(--foreground-muted))" }} axisLine={false} tickLine={false} />
                     <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "hsl(var(--foreground-muted))" }} axisLine={false} tickLine={false} />
                     <Tooltip 
                       contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }} 
                     />
                     
                     <Line yAxisId="left" type="monotone" dataKey="trainLoss" stroke="hsl(var(--brand))" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                     <Line yAxisId="left" type="monotone" dataKey="valLoss" stroke="hsl(var(--foreground-subtle))" strokeWidth={1.5} dot={false} strokeDasharray="5 5" isAnimationActive={false} />
                     <Line yAxisId="right" type="monotone" dataKey="acc" stroke="hsl(var(--level-low))" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                   </LineChart>
                 </ResponsiveContainer>
               </div>
               <div className="flex justify-center gap-6 mt-4 text-xs font-medium tracking-wider uppercase text-foreground-subtle">
                 <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-brand" /> Train Loss</div>
                 <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-foreground-subtle" /> Val Loss</div>
                 <div className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-level-low" /> Accuracy</div>
               </div>
            </Card>

            <div className="bg-background-muted border border-border rounded-lg p-4 font-mono text-xs text-foreground-muted h-48 overflow-y-auto">
              {isTraining ? (
                <>
                  <div className="text-foreground-subtle mb-2">Initializing compute worker thread...</div>
                  <div className="text-foreground-subtle mb-4">Allocated model: Gradient Bagging configured with 100 max estimators.</div>
                  {MOCK_HISTORY.map((h, i) => (
                    <div key={i} className="mb-1.5 flex gap-4">
                      <span className="text-brand">[{String(h.epoch).padStart(2, '0')}/100]</span>
                      <span>loss: <span className="text-foreground">{h.trainLoss.toFixed(4)}</span></span>
                      <span>val_loss: <span className="text-foreground">{h.valLoss.toFixed(4)}</span></span>
                      <span>acc: <span className="text-foreground">{h.acc.toFixed(2)}%</span></span>
                    </div>
                  ))}
                  <div className="animate-pulse text-brand mt-2">&gt;_ Processing...</div>
                </>
              ) : (
                <div className="flex items-center justify-center h-full opacity-50">
                   Await compute context...
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
               {(!isTraining) ? Array.from({length: 6}).map((_, i) => (
                 <div key={i} className="border border-border rounded-lg p-4 bg-surface opacity-50">
                   <div className="text-xs uppercase tracking-wider text-foreground-muted mb-2 animate-pulse h-3 w-16 bg-background-muted rounded" />
                   <div className="text-xl font-semibold tabular-nums text-foreground animate-pulse h-6 w-20 bg-background-muted rounded" />
                 </div>
               )) : (
                 <>
                   <div className="border border-border rounded-lg p-4 bg-surface">
                     <div className="text-xs uppercase tracking-wider text-foreground-muted mb-1">Peak Accuracy</div>
                     <div className="text-xl font-semibold tabular-nums text-foreground">87.50%</div>
                   </div>
                   <div className="border border-border rounded-lg p-4 bg-surface">
                     <div className="text-xs uppercase tracking-wider text-foreground-muted mb-1">Min Val Loss</div>
                     <div className="text-xl font-semibold tabular-nums text-foreground">0.1420</div>
                   </div>
                   <div className="border border-border rounded-lg p-4 bg-surface">
                     <div className="text-xs uppercase tracking-wider text-foreground-muted mb-1">Mean Time/Epoch</div>
                     <div className="text-xl font-semibold tabular-nums text-foreground">145ms</div>
                   </div>
                   <div className="border border-border rounded-lg p-4 bg-surface">
                     <div className="text-xs uppercase tracking-wider text-foreground-muted mb-1">Convergence</div>
                     <div className="text-xl font-semibold tabular-nums text-brand">Stable</div>
                   </div>
                   <div className="border border-border rounded-lg p-4 bg-surface hidden lg:block">
                     <div className="text-xs uppercase tracking-wider text-foreground-muted mb-1">Memory</div>
                     <div className="text-xl font-semibold tabular-nums text-foreground">1.2 GB</div>
                   </div>
                   <div className="border border-border rounded-lg p-4 bg-surface hidden lg:block">
                     <div className="text-xs uppercase tracking-wider text-foreground-muted mb-1">Compute Node</div>
                     <div className="text-xl font-semibold tabular-nums text-foreground">Local</div>
                   </div>
                 </>
               )}
            </div>

          </div>
        </div>
      </main>
    </div>
  )
}
