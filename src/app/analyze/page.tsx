"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/Button"
import { Badge } from "@/components/ui/Badge"
import { Input } from "@/components/ui/Input"
import { Upload, ChevronDown, ChevronRight, Loader2, CheckCircle2 } from "lucide-react"
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
  const [uploading, setUploading] = React.useState(false)
  const [uploadedData, setUploadedData] = React.useState<any>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)
  
  const [expanded, setExpanded] = React.useState({
    band: true,
    nonlinear: true,
    asymmetry: true
  })

  // Editable form state
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

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setUploading(true);
      
      const formData = new FormData();
      formData.append("file", selectedFile);
      
      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData
        });
        if (res.ok) {
          const parsed = await res.json();
          setUploadedData(parsed);
        } else {
          console.error("Upload failed");
        }
      } catch (err) {
        console.error("Upload error", err);
      } finally {
        setUploading(false);
      }
    }
  };

  const handleRunAnalysis = () => {
    let payload: any = {};
    if (activeTab === "upload") {
      if (uploadedData) {
        payload = {
          signals: uploadedData.signals,
          channels: uploadedData.channels,
          sampling_rate: uploadedData.sampling_rate,
          model: selectedModel,
          manual: false
        };
      } else {
        // Fallback mock continuous signals if no file uploaded
        const mockChannels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'];
        const mockSignals = mockChannels.map(() => 
          Array.from({ length: 3840 }, () => (Math.random() - 0.5) * 30)
        );
        payload = {
          signals: mockSignals,
          channels: mockChannels,
          sampling_rate: 128,
          model: selectedModel,
          manual: false
        };
      }
    } else {
      payload = {
        features: {
          "PSD_Delta_Fp1": parseFloat(features.delta) || 25.3,
          "PSD_Theta_Fp1": parseFloat(features.theta) || 12.8,
          "PSD_Alpha_Fp1": parseFloat(features.alpha) || 18.5,
          "PSD_Beta_Fp1": parseFloat(features.beta) || 15.2,
          "PSD_Gamma_Fp1": parseFloat(features.gamma) || 6.7,
          "PSD_Delta_F3": parseFloat(features.delta) || 25.3,
          "PSD_Theta_F3": parseFloat(features.theta) || 12.8,
          "PSD_Alpha_F3": parseFloat(features.f3_alpha) || 12.4,
          "PSD_Beta_F3": parseFloat(features.beta) || 15.2,
          "PSD_Gamma_F3": parseFloat(features.gamma) || 6.7,
          "PSD_Delta_F4": parseFloat(features.delta) || 25.3,
          "PSD_Theta_F4": parseFloat(features.theta) || 12.8,
          "PSD_Alpha_F4": parseFloat(features.f4_alpha) || 15.1,
          "PSD_Beta_F4": parseFloat(features.beta) || 15.2,
          "PSD_Gamma_F4": parseFloat(features.gamma) || 6.7
        },
        model: selectedModel,
        manual: true
      };
    }

    try {
      localStorage.setItem("neuriq-analysis-request", JSON.stringify(payload));
    } catch (e) {
      console.error("Failed to save request payload to localStorage", e);
    }
    
    router.push("/results");
  };

  const loadPreset = (type: "anxious" | "relaxed") => {
    if (type === "anxious") {
      setFeatures({
        delta: "14.2", theta: "8.5", alpha: "5.8", beta: "28.4", gamma: "11.2",
        hjorth_activity: "1.4", sample_entropy: "0.65", hjorth_complexity: "1.25",
        f3_alpha: "5.2", f4_alpha: "11.4"
      });
    } else {
      setFeatures({
        delta: "25.3", theta: "12.8", alpha: "18.5", beta: "15.2", gamma: "6.7",
        hjorth_activity: "0.8", sample_entropy: "1.2", hjorth_complexity: "2.1",
        f3_alpha: "12.4", f4_alpha: "15.1"
      });
    }
  };

  const clearAll = () => {
    setFeatures({
      delta: "", theta: "", alpha: "", beta: "", gamma: "",
      hjorth_activity: "", sample_entropy: "", hjorth_complexity: "",
      f3_alpha: "", f4_alpha: ""
    });
  };

  // Re-map the signal preview graph dynamically based on uploaded data
  const waveformData = React.useMemo(() => {
    if (uploadedData && uploadedData.signals && uploadedData.signals.length > 0) {
      const length = Math.min(50, uploadedData.signals[0].length);
      const ch1 = uploadedData.channels[0] || "Ch1";
      const ch2 = uploadedData.channels[1] || "Ch2";
      return Array.from({ length }, (_, i) => ({
        time: i,
        [ch1]: uploadedData.signals[0][i],
        [ch2]: uploadedData.signals[1] ? uploadedData.signals[1][i] : 0
      }));
    }
    return MOCK_WAVEFORM;
  }, [uploadedData]);

  const SectionHeader = ({ title, id }: { title: string, id: keyof typeof expanded }) => (
    <div 
      className="flex items-center justify-between py-3 border-b border-border cursor-pointer mb-4"
      onClick={() => setExpanded(p => ({ ...p, [id]: !p[id] }))}
    >
      <span className="text-sm font-medium text-foreground">{title}</span>
      {expanded[id] ? <ChevronDown className="w-4 h-4 text-foreground-muted" /> : <ChevronRight className="w-4 h-4 text-foreground-muted" />}
    </div>
  )

  const activeChannel1 = uploadedData?.channels?.[0] || "AF3";
  const activeChannel2 = uploadedData?.channels?.[1] || "F7";

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
              className="border-2 border-dashed border-border rounded-lg p-12 text-center hover:border-brand hover:bg-brand/5 transition-colors cursor-pointer group relative"
              onClick={() => !uploading && fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                className="hidden" 
                ref={fileInputRef} 
                accept=".edf,.bdf,.mat,.csv"
                onChange={handleFileChange}
                disabled={uploading}
              />
              {uploading ? (
                <div className="py-4 flex flex-col items-center">
                  <Loader2 className="animate-spin text-brand w-8 h-8 mb-3" />
                  <h3 className="text-sm font-medium text-foreground">Processing EEG Channels...</h3>
                  <p className="text-xs text-foreground-muted mt-1">Filtering signals and calculating spectral attributes</p>
                </div>
              ) : uploadedData ? (
                <div className="py-2">
                  <CheckCircle2 className="mx-auto text-level-low w-8 h-8 mb-3" />
                  <h3 className="text-sm font-medium text-foreground">{file?.name}</h3>
                  <p className="text-xs text-brand font-medium mt-1">
                    Successfully loaded {uploadedData.channels.length} EEG channels ({uploadedData.duration_sec}s @ {uploadedData.sampling_rate}Hz)
                  </p>
                  <p className="text-xs text-foreground-subtle mt-2">Click to replace file</p>
                </div>
              ) : (
                <>
                  <Upload className="mx-auto text-foreground-subtle group-hover:text-brand transition-colors w-6 h-6" />
                  <h3 className="text-sm font-medium text-foreground mt-3">
                    Drag & drop EEG recording
                  </h3>
                  <p className="text-xs text-foreground-muted mt-1">
                    Accepted formats: .edf, .bdf, .mat, .csv
                  </p>
                </>
              )}
            </div>
          ) : (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
              
              <div>
                <SectionHeader title="EEG Band Powers" id="band" />
                <AnimatePresence>
                  {expanded.band && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="grid grid-cols-2 gap-4">
                      <Input label="Delta" unit="µV²" value={features.delta} onChange={(e) => setFeatures(p => ({ ...p, delta: e.target.value }))} />
                      <Input label="Theta" unit="µV²" value={features.theta} onChange={(e) => setFeatures(p => ({ ...p, theta: e.target.value }))} />
                      <Input label="Alpha" unit="µV²" value={features.alpha} onChange={(e) => setFeatures(p => ({ ...p, alpha: e.target.value }))} />
                      <Input label="Beta" unit="µV²" value={features.beta} onChange={(e) => setFeatures(p => ({ ...p, beta: e.target.value }))} />
                      <Input label="Gamma" unit="µV²" value={features.gamma} onChange={(e) => setFeatures(p => ({ ...p, gamma: e.target.value }))} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div>
                <SectionHeader title="Nonlinear Features" id="nonlinear" />
                <AnimatePresence>
                  {expanded.nonlinear && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="grid grid-cols-2 gap-4">
                      <Input label="Hjorth Activity" value={features.hjorth_activity} onChange={(e) => setFeatures(p => ({ ...p, hjorth_activity: e.target.value }))} />
                      <Input label="Sample Entropy" value={features.sample_entropy} onChange={(e) => setFeatures(p => ({ ...p, sample_entropy: e.target.value }))} />
                      <Input label="Hjorth Complexity" value={features.hjorth_complexity} onChange={(e) => setFeatures(p => ({ ...p, hjorth_complexity: e.target.value }))} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div>
                <SectionHeader title="Frontal Asymmetry" id="asymmetry" />
                <AnimatePresence>
                  {expanded.asymmetry && (
                    <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="grid grid-cols-2 gap-4">
                      <Input label="F3 Alpha Power" unit="dB" value={features.f3_alpha} onChange={(e) => setFeatures(p => ({ ...p, f3_alpha: e.target.value }))} />
                      <Input label="F4 Alpha Power" unit="dB" value={features.f4_alpha} onChange={(e) => setFeatures(p => ({ ...p, f4_alpha: e.target.value }))} />
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              <div className="flex gap-2 mt-6">
                <Button variant="outline" size="sm" onClick={() => loadPreset("anxious")}>Anxious Example</Button>
                <Button variant="outline" size="sm" onClick={() => loadPreset("relaxed")}>Non-Anxious Example</Button>
                <Button variant="ghost" size="sm" className="ml-auto" onClick={clearAll}>Clear All</Button>
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
            <Button size="lg" className="w-full" onClick={handleRunAnalysis} disabled={uploading}>
              Run Anxiety Analysis
            </Button>
            <p className="text-xs text-foreground-subtle text-center mt-3">Estimated time: under 2 seconds</p>
          </div>

        </div>

        {/* RIGHT PANEL */}
        <div className="col-span-2 sticky top-20">
          <h3 className="text-sm font-medium text-foreground mb-4">Signal Preview</h3>
          
          <div className="bg-background-subtle border border-border rounded-lg p-4 mb-6">
            <div className="h-[200px] w-full mb-3">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformData} margin={{ top: 5, right: 0, left: -20, bottom: 0 }}>
                  <XAxis dataKey="time" hide />
                  <YAxis hide domain={['auto', 'auto']} />
                  <Line type="monotone" dataKey={activeChannel1} stroke="hsl(var(--brand))" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                  <Line type="monotone" dataKey={activeChannel2} stroke="hsl(var(--border-strong))" strokeWidth={1} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            
            <div className="grid grid-cols-7 gap-1">
              {(uploadedData?.channels?.slice(0, 7) || ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"]).map((ch: string, i: number) => (
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
                  "Alpha Power": `${parseFloat(features.alpha || "0").toFixed(1)} µV²`,
                  "Beta Power": `${parseFloat(features.beta || "0").toFixed(1)} µV²`,
                  "Alpha/Beta Ratio": features.alpha && features.beta ? (parseFloat(features.alpha) / parseFloat(features.beta)).toFixed(2) : "1.22",
                  "Frontal Asymmetry": features.f4_alpha && features.f3_alpha ? (parseFloat(features.f4_alpha) - parseFloat(features.f3_alpha)).toFixed(2) : "-0.23",
                  "Sample Entropy": `${parseFloat(features.sample_entropy || "0").toFixed(2)}`
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
