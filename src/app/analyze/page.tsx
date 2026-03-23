"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/Button"
import { Badge } from "@/components/ui/Badge"
import { Input } from "@/components/ui/Input"
import { Upload, ChevronDown, ChevronRight } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { useRouter } from "next/navigation"

const MOCK_WAVEFORM = Array.from({ length: 50 }, (_, i) => ({
  time: i,
  AF3: Math.sin(i * 0.2) * 40 + Math.random() * 10,
  F7: Math.cos(i * 0.15) * 35 + Math.random() * 8,
}))

export default function AnalyzePage() {
  const router = useRouter()
  const [activeTab, setActiveTab] = React.useState<"upload" | "manual">("upload")
  const [selectedModel, setSelectedModel] = React.useState("gradient")
  const [file, setFile] = React.useState<File | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  
  const [expanded, setExpanded] = React.useState({
    band: true,
    nonlinear: true,
    asymmetry: true
  })

  // Mock form state
  const [features, setFeatures] = React.useState({
    delta: "25.3", theta: "12.8", alpha: "18.5", beta: "15.2", gamma: "6.7",
    hjorth_activity: "0.8", sample_entropy: "1.2", hjorth_complexity: "2.1",
    f3_alpha: "12.4", f4_alpha: "15.1"
  })

  const models = [
    { id: "random_forest", name: "Random Forest", acc: "86.1%", desc: "Baseline decision trees" },
    { id: "gradient", name: "Gradient Bagging", acc: "87.5%", desc: "Top performing ensemble" },
    { id: "svm", name: "SVM", acc: "82.4%", desc: "Linear separation margin" },
    { id: "cnn", name: "CNN-LSTM", acc: "89.2%", desc: "Deep spatio-temporal" }
  ]

  const SectionHeader = ({ title, id }: { title: string, id: keyof typeof expanded }) => (
    <div 
      className="flex items-center justify-between py-3 border-b border-border cursor-pointer mb-4"
      onClick={() => setExpanded(p => ({ ...p, [id]: !p[id] }))}
    >
      <span className="text-sm font-medium text-foreground">{title}</span>
      {expanded[id] ? <ChevronDown className="w-4 h-4 text-foreground-muted" /> : <ChevronRight className="w-4 h-4 text-foreground-muted" />}
    </div>
  )

  return (
    <div className="w-full pt-16 pb-24 px-0 max-w-7xl mx-auto">
      {/* PAGE HEADER */}
      <div className="mb-8 border-b border-border pb-8">
        <div className="text-xs text-foreground-subtle mb-2">Dashboard / Analysis</div>
        <h1 className="text-3xl font-semibold text-foreground tracking-tight">Signal Analysis</h1>
        <p className="text-sm text-foreground-muted mt-1 max-w-xl">
          Process raw EEG files or simulate manual features for instantaneous inference.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-12 items-start">
        {/* LEFT PANEL */}
        <div className="col-span-3 space-y-8">
          
          <div className="flex border-b border-border">
            <button 
              onClick={() => setActiveTab("upload")}
              className={`pb-3 text-sm font-medium transition-colors border-b-2 mr-6 ${activeTab === "upload" ? "border-brand text-foreground" : "border-transparent text-foreground-muted hover:text-foreground"}`}
            >
              Upload File
            </button>
            <button 
              onClick={() => setActiveTab("manual")}
              className={`pb-3 text-sm font-medium transition-colors border-b-2 ${activeTab === "manual" ? "border-brand text-foreground" : "border-transparent text-foreground-muted hover:text-foreground"}`}
            >
              Manual Input
            </button>
          </div>

          {activeTab === "upload" ? (
            <div 
              className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-brand hover:bg-brand/5 transition-colors cursor-pointer group"
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                className="hidden" 
                ref={fileInputRef} 
                accept=".edf,.bdf,.mat,.csv"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setFile(e.target.files[0])
                  }
                }}
              />
              <Upload className="mx-auto text-foreground-subtle group-hover:text-brand transition-colors w-6 h-6" />
              <h3 className="text-sm font-medium text-foreground mt-3">
                {file ? file.name : "Drag & drop EEG recording"}
              </h3>
              <p className="text-xs text-foreground-muted mt-1">
                {file ? `${(file.size / 1024).toFixed(1)} KB` : "Accepted formats: .edf, .bdf, .mat, .csv"}
              </p>
            </div>
          ) : (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              
              <div>
                <SectionHeader title="EEG Band Powers" id="band" />
                <AnimatePresence>
                  {expanded.band && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="grid grid-cols-2 gap-4">
                      <Input label="Delta" unit="µV²" value={features.delta} readOnly />
                      <Input label="Theta" unit="µV²" value={features.theta} readOnly />
                      <Input label="Alpha" unit="µV²" value={features.alpha} readOnly />
                      <Input label="Beta" unit="µV²" value={features.beta} readOnly />
                      <Input label="Gamma" unit="µV²" value={features.gamma} readOnly />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div>
                <SectionHeader title="Nonlinear Features" id="nonlinear" />
                <AnimatePresence>
                  {expanded.nonlinear && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="grid grid-cols-2 gap-4">
                      <Input label="Hjorth Activity" value={features.hjorth_activity} readOnly />
                      <Input label="Sample Entropy" value={features.sample_entropy} readOnly />
                      <Input label="Hjorth Complexity" value={features.hjorth_complexity} readOnly />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div>
                <SectionHeader title="Frontal Asymmetry" id="asymmetry" />
                <AnimatePresence>
                  {expanded.asymmetry && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="grid grid-cols-2 gap-4">
                      <Input label="F3 Alpha Power" unit="dB" value={features.f3_alpha} readOnly />
                      <Input label="F4 Alpha Power" unit="dB" value={features.f4_alpha} readOnly />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div className="flex gap-2 mt-6">
                <Button variant="outline" size="sm">Anxious Example</Button>
                <Button variant="outline" size="sm">Non-Anxious Example</Button>
                <Button variant="ghost" size="sm" className="ml-auto">Clear All</Button>
              </div>
            </motion.div>
          )}

          <div className="pt-6 border-t border-border">
            <h3 className="text-xs font-medium text-foreground-muted uppercase tracking-wider mb-3">Target Model</h3>
            <div className="grid grid-cols-2 gap-4">
              {models.map(m => (
                <div 
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  className={`relative border rounded-lg p-4 cursor-pointer transition-colors ${selectedModel === m.id ? 'border-brand bg-brand/5' : 'border-border bg-surface hover:border-border-strong'}`}
                >
                  <div className="absolute top-4 right-4"><Badge variant="default" className="bg-level-low-bg text-level-low">{m.acc}</Badge></div>
                  <h4 className="text-sm font-medium text-foreground">{m.name}</h4>
                  <p className="text-xs text-foreground-muted mt-0.5">{m.desc}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-4">
            <Button size="lg" className="w-full" onClick={() => router.push('/results')}>Run Anxiety Analysis</Button>
            <p className="text-xs text-foreground-subtle text-center mt-3">Estimated time: under 2 seconds</p>
          </div>

        </div>

        {/* RIGHT PANEL */}
        <div className="col-span-2 sticky top-20">
          <h3 className="text-sm font-medium text-foreground mb-4">Signal Preview</h3>
          
          <div className="bg-background-subtle border border-border rounded-lg p-4 mb-6">
            <div className="h-[200px] w-full mb-3">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={MOCK_WAVEFORM} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                  <XAxis dataKey="time" hide />
                  <YAxis hide domain={['auto', 'auto']} />
                  <Line type="monotone" dataKey="AF3" stroke="hsl(var(--brand))" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey="F7" stroke="hsl(var(--border-strong))" strokeWidth={1} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="grid grid-cols-7 gap-1">
              {["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"].map((ch, i) => (
                <div key={ch} className={`text-xs border rounded px-2 py-0.5 text-center cursor-pointer ${i === 0 ? "bg-foreground text-background border-foreground" : "border-border text-foreground-subtle hover:text-foreground"}`}>
                  {ch}
                </div>
              ))}
            </div>
          </div>

          <h3 className="text-sm font-medium text-foreground mb-4">Feature Summary</h3>
          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-left text-xs">
              <tbody>
                {Object.entries({
                  "Alpha Power": "18.5 µV²",
                  "Beta Power": "15.2 µV²",
                  "Alpha/Beta Ratio": "1.22",
                  "Frontal Asymmetry": "-0.23",
                  "Sample Entropy": "1.20"
                }).map(([k, v], i) => (
                  <tr key={k} className={i % 2 === 0 ? "bg-background-subtle/50" : "bg-transparent"}>
                    <td className="px-4 py-2 text-foreground-muted">{k}</td>
                    <td className="px-4 py-2 font-medium text-foreground text-right">{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}
