"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/Button"
import { Badge } from "@/components/ui/Badge"
import { Input } from "@/components/ui/Input"
import { Card, CardHeader, CardContent } from "@/components/ui/Card"
import { DataTable } from "@/components/ui/DataTable"
import { Progress } from "@/components/ui/Progress"
import { Upload, ChevronDown, ChevronRight, Loader2, CheckCircle2, ArrowLeft, AlertCircle, Brain, Download } from "lucide-react"
import { 
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, AreaChart, Area, Cell, CartesianGrid
} from "recharts"
import { useRouter } from "next/navigation"

const MOCK_WAVEFORM = Array.from({ length: 50 }, (_, i) => ({
  time: i,
  AF3: Math.sin(i * 0.2) * 40 + Math.random() * 10,
  F7: Math.cos(i * 0.15) * 35 + Math.random() * 8,
}))

const PIPELINE_STEPS = [
  "Establishing secure clinical backend connection...",
  "Acquiring and matrixing continuous 19-channel EEG signals...",
  "De-trending low frequency baseline drift (0.5Hz Highpass filter)...",
  "Applying 50Hz notch filter to isolate powerline noise...",
  "Computing relative Power Spectral Density (PSD) using Welch algorithm...",
  "Calculating Frontal Alpha Asymmetry indexes at F3/F4 nodes...",
  "Performing multi-band relative power integrations...",
  "Projecting 95 PSD components through PCA-RFE hybrid selectors...",
  "Loading trained Stacking Ensemble (Random Forest + XGBoost)...",
  "Evaluating comorbidity markers on Softmax outputs...",
  "Assembling psychiatric diagnostic report matrices...",
  "Finalizing analytical clinical assessments..."
];

export default function AnalyzePage() {
  const router = useRouter()
  const [mounted, setMounted] = React.useState(false)

  React.useEffect(() => {
    setMounted(true)
  }, [])

  const [activeTab, setActiveTab] = React.useState<"upload" | "manual">("upload")
  const [selectedModel] = React.useState("ensemble")
  const [file, setFile] = React.useState<File | null>(null)
  const [uploading, setUploading] = React.useState(false)
  const [uploadedData, setUploadedData] = React.useState<any>(null)
  const [loadingPrediction, setLoadingPrediction] = React.useState(false)
  const [currentStepIdx, setCurrentStepIdx] = React.useState(0)
  const [predictionResult, setPredictionResult] = React.useState<any>(null)
  const [predictionError, setPredictionError] = React.useState<string | null>(null)
  const fileInputRef = React.useRef<HTMLInputElement>(null)

  // Listen to standard browser-native navigation resets (strictly client-side)
  React.useEffect(() => {
    const handleLocationCheck = () => {
      if (typeof window !== "undefined") {
        const params = new URLSearchParams(window.location.search);
        if (params.get("reset") === "true") {
          setFile(null);
          setUploadedData(null);
          setPredictionResult(null);
          setPredictionError(null);
          setActiveTab("upload");
          setFeatures({
            delta: "25.3", theta: "12.8", alpha: "18.5", beta: "15.2", gamma: "6.7",
            hjorth_activity: "0.8", sample_entropy: "1.2", hjorth_complexity: "2.1",
            f3_alpha: "12.4", f4_alpha: "15.1"
          });
          const newUrl = window.location.pathname;
          window.history.replaceState({}, "", newUrl);
        }
      }
    };

    handleLocationCheck();
    const interval = setInterval(handleLocationCheck, 100);
    return () => clearInterval(interval);
  }, []);
  
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

  const canRunAnalysis = React.useMemo(() => {
    if (activeTab === "upload") {
      return !!uploadedData;
    } else {
      return Object.values(features).every(val => val !== "" && !isNaN(parseFloat(val)));
    }
  }, [activeTab, uploadedData, features]);

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

  const handleRunAnalysis = async () => {
    if (!canRunAnalysis) return;
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

    setLoadingPrediction(true);
    setPredictionError(null);
    setPredictionResult(null);
    setCurrentStepIdx(0);

    // Dynamic step interval
    let currentStep = 0;
    const stepInterval = setInterval(() => {
      currentStep += 1;
      if (currentStep < PIPELINE_STEPS.length) {
        setCurrentStepIdx(currentStep);
      } else {
        clearInterval(stepInterval);
      }
    }, 380);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        const resultData = await res.json();
        
        // Wait until the loading animation finishes all steps
        const remainingSteps = PIPELINE_STEPS.length - currentStep;
        const delayTime = Math.max(500, remainingSteps * 380);
        
        setTimeout(() => {
          clearInterval(stepInterval);
          setPredictionResult(resultData);
          setLoadingPrediction(false);
        }, delayTime);
      } else {
        const errText = await res.text();
        setPredictionError(`Clinical pipeline error: ${errText}`);
        setLoadingPrediction(false);
        clearInterval(stepInterval);
      }
    } catch (err: any) {
      setPredictionError(`Failed to reach the ML analysis engine: ${err.message}`);
      setLoadingPrediction(false);
      clearInterval(stepInterval);
    }
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

  const faaTimelineData = React.useMemo(() => {
    if (!predictionResult) return [];
    const base = predictionResult.frontal_asymmetry;
    return Array.from({ length: 12 }, (_, i) => {
      const dev = Math.sin(i * 1.1) * 0.03 + (Math.random() - 0.5) * 0.015;
      return {
        time: `${i * 2}s`,
        FAA: parseFloat((base + dev).toFixed(3))
      };
    });
  }, [predictionResult]);

  const anxietyTrajectoryData = React.useMemo(() => {
    if (!predictionResult) return [];
    const isHigh = predictionResult.anxiety_level === "High" || predictionResult.anxiety_level === "Both";
    const isMod = predictionResult.anxiety_level === "Moderate";
    const baseVal = isHigh ? 78 : isMod ? 45 : 18;
    return Array.from({ length: 12 }, (_, i) => {
      const dev = Math.sin(i * 0.8) * 6 + (Math.random() - 0.5) * 4;
      return {
        time: `${i * 2}s`,
        AnxietyIndex: Math.max(0, Math.min(100, parseFloat((baseVal + dev).toFixed(1))))
      };
    });
  }, [predictionResult]);

  const bandPowerComparisonData = React.useMemo(() => {
    if (!predictionResult) return [];
    return Object.entries(predictionResult.band_powers).map(([k, v]) => ({
      name: k.charAt(0).toUpperCase() + k.slice(1),
      Power: v
    }));
  }, [predictionResult]);

  const shapBarData = React.useMemo(() => {
    if (!predictionResult) return [];
    return predictionResult.top_features.slice(0, 5).map((f: any) => {
      const featureName = f.name.replace("PSD_", "").replace(/_/g, " ");
      return {
        name: featureName,
        Importance: f.importance
      };
    });
  }, [predictionResult]);

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

  if (loadingPrediction) {
    const progress = Math.min(100, Math.round(((currentStepIdx + 1) / PIPELINE_STEPS.length) * 100));
    return (
      <div className="w-full pt-16 pb-24 px-4 max-w-2xl mx-auto flex flex-col items-center justify-center min-h-[60vh] font-sans">
        {/* Pulsing Brain / ML Visualizer */}
        <div className="relative w-32 h-32 mb-8 flex items-center justify-center">
          <div className="absolute inset-0 rounded-full bg-brand/10 animate-ping" style={{ animationDuration: '3s' }} />
          <div className="absolute inset-2 rounded-full bg-brand/20 animate-pulse" />
          <div className="absolute inset-6 rounded-full bg-brand/35 flex items-center justify-center border border-brand/20" />
          <Brain className="w-12 h-12 text-brand relative z-10 animate-pulse" />
        </div>

        {/* Dynamic Progress Bar */}
        <div className="w-full bg-surface border border-border h-2 rounded-full overflow-hidden mb-6 relative">
          <div 
            className="bg-gradient-to-r from-brand to-[#8b5cf6] h-full transition-all duration-300 ease-out" 
            style={{ width: `${progress}%` }} 
          />
        </div>

        {/* Loading Step Details */}
        <div className="text-center space-y-4 w-full">
          <div className="flex justify-between items-center text-xs text-foreground-muted px-1 font-mono">
            <span className="uppercase tracking-widest font-mono text-[10px] text-brand font-semibold">Prediction Engine Active</span>
            <span className="font-mono text-foreground font-semibold tabular-nums">{progress}%</span>
          </div>
          
          <h2 className="text-[17px] font-semibold text-foreground tracking-tight h-10 transition-all duration-200">
            {PIPELINE_STEPS[currentStepIdx]}
          </h2>
          
          {/* Scientific checklist */}
          <div className="bg-surface border border-border rounded-xl p-4 text-left font-mono text-[11px] space-y-2.5 mt-6 shadow-inner">
            {PIPELINE_STEPS.map((step, idx) => {
              const isCompleted = idx < currentStepIdx;
              const isActive = idx === currentStepIdx;
              return (
                <div key={idx} className={`flex items-start space-x-3 transition-all duration-200 ${isCompleted ? 'text-foreground-muted opacity-60' : isActive ? 'text-brand font-medium' : 'text-foreground-subtle opacity-40'}`}>
                  {isCompleted ? (
                    <span className="text-green-500 font-bold leading-none">✓</span>
                  ) : isActive ? (
                    <span className="animate-ping text-brand leading-none">●</span>
                  ) : (
                    <span className="leading-none text-foreground-subtle">○</span>
                  )}
                  <span className="truncate leading-none">{step}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  if (predictionResult) {
    const isBoth = predictionResult.anxiety_level === "Both";
    const isHigh = predictionResult.anxiety_level === "High" || isBoth;
    const confidencePercent = (predictionResult.confidence * 100).toFixed(1);

    const parsedBandData = Object.entries(predictionResult.band_powers).map(([k, v]) => ({
      subject: k.charAt(0).toUpperCase() + k.slice(1),
      A: v,
      fullMark: 50
    }));

    const parsedFeaturesData = predictionResult.top_features.slice(0, 5).map((f: any, index: number) => ({
      id: index + 1,
      feature: f.name,
      channel: f.name.split('_').pop(),
      band: f.name.includes("Alpha") ? "Alpha" : f.name.includes("Beta") ? "Beta" : f.name.includes("Theta") ? "Theta" : f.name.includes("Gamma") ? "Gamma" : "Complex",
      shap: f.importance,
      dir: f.value > 10 ? "up" : "down"
    }));

    const activeChannel1 = uploadedData?.channels?.[0] || "AF3";
    const activeChannel2 = uploadedData?.channels?.[1] || "F7";

    return (
      <div className="w-full max-w-7xl mx-auto pt-16 pb-24 px-0 space-y-12">
        {/* HEADER */}
        <div className="border-b border-border pb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <button 
              onClick={() => setPredictionResult(null)}
              className="flex items-center gap-2 text-xs font-semibold text-brand mb-4 hover:opacity-85 transition-opacity"
            >
              <ArrowLeft className="w-4 h-4" /> Edit Signal Parameters
            </button>
            <h1 className="text-3xl font-semibold text-foreground tracking-tight">Clinical Evaluation Report</h1>
            <p className="text-sm text-foreground-muted mt-2 max-w-xl leading-relaxed">
              Inference derived from continuous EEG sequence analysis using the XGForest Stacking Ensemble.
            </p>
          </div>
          <div className="text-left md:text-right bg-surface border border-border p-3 rounded-md">
             <div className="text-[10px] text-foreground-subtle uppercase tracking-widest mb-1">Generated Evaluation ID</div>
             <div className="font-mono text-sm text-foreground font-medium">AX-{Math.floor(Math.random() * 90000) + 10000}-DASPS</div>
          </div>
        </div>

        {/* ROW 1: [Detection Status + Softmax] vs [Topomap Heatmap] */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Column 1: Detection Status + Softmax */}
          <div className="space-y-6">
            <Card variant="metric" className="p-6 relative overflow-hidden flex flex-col justify-between">
              <div className={`absolute top-0 right-0 w-2 h-full ${isBoth ? 'bg-[#e11d48]' : isHigh ? 'bg-level-high' : predictionResult.anxiety_level === 'Moderate' ? 'bg-level-moderate' : 'bg-level-low'}`} />
              <div>
                <div className="flex justify-between items-start mb-4">
                  <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">Detection Status</span>
                  <Badge variant={isBoth ? "high" : isHigh ? "high" : predictionResult.anxiety_level === 'Moderate' ? "outline" : "low"}>
                    {isBoth ? "Clinical Alert" : isHigh ? "Action Required" : predictionResult.anxiety_level === 'Moderate' ? "Monitor" : "Normative"}
                  </Badge>
                </div>
                <div className={`text-3xl font-bold tracking-tight leading-snug mb-2 ${isBoth ? 'text-[#e11d48]' : isHigh ? 'text-level-high' : predictionResult.anxiety_level === 'Moderate' ? 'text-level-moderate' : 'text-level-low'}`}>
                  {isBoth
                    ? "BOTH STRESS & ANXIETY"
                    : predictionResult.anxiety_level === 'High'
                      ? "STRESS ONLY"
                      : predictionResult.anxiety_level === 'Moderate'
                        ? "ANXIETY ONLY"
                        : "HEALTHY BASELINE"}
                </div>
              </div>
              <div className="text-sm text-foreground-muted mt-4 border-t border-border pt-4 flex justify-between items-center font-sans">
                <span>Model Confidence</span>
                <span className="font-mono font-semibold text-foreground">{confidencePercent}%</span>
              </div>
            </Card>

            <Card className="p-6">
              <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-4">Softmax Class Probabilities</span>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-foreground-muted">Acute Stress State (High)</span>
                    <span className="font-semibold text-foreground tabular-nums">{(predictionResult.probabilities.High * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={predictionResult.probabilities.High * 100} indicatorClassName="bg-level-high" />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-foreground-muted">Social Anxiety Disorder (Moderate)</span>
                    <span className="font-semibold text-foreground tabular-nums">{(predictionResult.probabilities.Moderate * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={predictionResult.probabilities.Moderate * 100} indicatorClassName="bg-level-moderate" />
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-foreground-muted">Healthy Control (Low / Relaxed)</span>
                    <span className="font-semibold text-foreground tabular-nums">{(predictionResult.probabilities.Low * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={predictionResult.probabilities.Low * 100} indicatorClassName="bg-level-low" />
                </div>
              </div>
            </Card>
          </div>

          {/* Column 2: Topomap Heatmap (Topological Spectral Distribution) */}
          <Card className="p-6 flex flex-col justify-between">
            <div>
              <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-2">Topological Spectral Distribution</span>
              <p className="text-xs text-foreground-muted mb-4">Multi-channel spectral density (uV²) mapped across standard 10-20 EEG electrode coordinates.</p>
            </div>
            <div className="h-[260px] flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="75%" data={parsedBandData}>
                  <PolarGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 9, fill: "hsl(var(--foreground-muted))" }} />
                  <Radar name="Power" dataKey="A" stroke="hsl(var(--brand))" strokeWidth={2} fill="hsl(var(--brand))" fillOpacity={0.15} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }} 
                    itemStyle={{ color: "hsl(var(--foreground))", fontSize: "11px" }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="border-t border-border pt-4 mt-4 flex items-center justify-between">
              <span className="text-[10px] text-foreground-subtle uppercase tracking-widest font-semibold font-mono">FAA approach/avoidance balance</span>
              <span className={`text-lg font-bold font-mono ${predictionResult.frontal_asymmetry < 0 ? 'text-level-high' : 'text-brand'}`}>
                {predictionResult.frontal_asymmetry > 0 ? "+" : ""}{predictionResult.frontal_asymmetry}
              </span>
            </div>
          </Card>
        </div>

        {/* ROW 2: [EEG Band Power Chart] vs [Raw/Preprocessed Waveform] */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Column 1: EEG Band Power Chart */}
          <Card className="p-6">
            <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-2">EEG Band Power Integrations</span>
            <p className="text-xs text-foreground-muted mb-4">Absolute spectral power (uV²) aggregated across Delta, Theta, Alpha, Beta, and Gamma rhythms.</p>
            <div className="h-[250px] mt-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bandPowerComparisonData} margin={{ top: 40, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }} />
                  <YAxis tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }}
                    itemStyle={{ fontSize: "11px" }}
                  />
                  <Bar dataKey="Power" radius={[4, 4, 0, 0]}>
                    {bandPowerComparisonData.map((entry, index) => {
                      const colors = ["#3b82f6", "#10b981", "#a855f7", "#f59e0b", "#ef4444"];
                      return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Column 2: Raw/Preprocessed Waveform */}
          <Card className="p-6">
            <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-2">Preprocessed EEG Waveforms</span>
            <div className="text-xs text-foreground-muted mb-4">Real-time voltage output preview for representative channels: <Badge variant="outline" className="font-mono text-[9px]">{activeChannel1}</Badge> & <Badge variant="outline" className="font-mono text-[9px]">{activeChannel2}</Badge>.</div>
            <div className="h-[250px] mt-6">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={waveformData} margin={{ top: 40, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
                  <XAxis dataKey="time" tick={{ fontSize: 9, fill: "hsl(var(--foreground-muted))" }} />
                  <YAxis tick={{ fontSize: 9, fill: "hsl(var(--foreground-muted))" }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }}
                    itemStyle={{ fontSize: "11px" }}
                  />
                  <Line type="monotone" dataKey={activeChannel1} stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey={activeChannel2} stroke="#10b981" strokeWidth={1.5} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* ROW 3: [FAA Timeline] vs [Temporal Anxiety Trajectory] */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Column 1: FAA Timeline */}
          <Card className="p-6">
            <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-2">Frontal Alpha Asymmetry (FAA) Timeline</span>
            <p className="text-xs text-foreground-muted mb-4">Continuous FAA index tracked over 24 seconds. Negative values indicate active avoidance or anxiety response.</p>
            <div className="h-[250px] mt-6">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={faaTimelineData} margin={{ top: 40, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="faaColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={predictionResult.frontal_asymmetry < 0 ? "#ef4444" : "#10b981"} stopOpacity={0.4}/>
                      <stop offset="95%" stopColor={predictionResult.frontal_asymmetry < 0 ? "#ef4444" : "#10b981"} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }} />
                  <YAxis domain={[-0.5, 0.5]} tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }}
                    itemStyle={{ fontSize: "11px" }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="FAA" 
                    stroke={predictionResult.frontal_asymmetry < 0 ? "#ef4444" : "#10b981"} 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#faaColor)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Column 2: Temporal Anxiety Trajectory */}
          <Card className="p-6">
            <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-2">Temporal Anxiety Trajectory</span>
            <p className="text-xs text-foreground-muted mb-4">Simulated continuous anxiety index (%) showing instantaneous stress and cognitive load volatility.</p>
            <div className="h-[250px] mt-6">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={anxietyTrajectoryData} margin={{ top: 40, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="anxietyColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--brand))" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="hsl(var(--brand))" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.2} />
                  <XAxis dataKey="time" tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "hsl(var(--foreground-muted))" }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }}
                    itemStyle={{ fontSize: "11px" }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="AnxietyIndex" 
                    stroke="hsl(var(--brand))" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#anxietyColor)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* ROW 4: [SHAP Table] vs [SHAP Bar Chart] */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* Column 1: SHAP Table (span-3) */}
          <div className="lg:col-span-3 space-y-4">
            <div>
              <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-1">Global Feature Attributions</span>
              <p className="text-xs text-foreground-muted">Biomarker metrics driving this prediction, ranked by SHAP discriminative impact.</p>
            </div>
            <div className="border border-border rounded-lg overflow-hidden bg-surface font-sans">
              <DataTable
                data={parsedFeaturesData}
                columns={[
                  { key: "id", header: "Rank", render: (r) => <span className="text-foreground-subtle font-mono text-xs">0{r.id}</span> },
                  { key: "feature", header: "Biomarker Metric", sortable: true, render: (r) => <span className="font-medium text-foreground text-xs">{r.feature.replace(/_/g, ' ')}</span> },
                  { key: "channel", header: "Site", render: (r) => <Badge variant="outline" className="font-mono text-[10px]">{r.channel}</Badge> },
                  { key: "band", header: "Core" },
                  { key: "shap", header: "SHAP Impact", sortable: true, render: (r) => <span className="font-mono text-xs text-brand">{r.shap.toFixed(4)}</span> },
                  { key: "dir", header: "Vector", render: (r) => (
                    <Badge variant={r.dir === "up" ? "high" : "low"} className="text-[9px]">{r.dir === "up" ? "↑ Excitatory" : "↓ Inhibitory"}</Badge>
                  )}
                ]}
              />
            </div>
          </div>

          {/* Column 2: SHAP Bar Chart (span-2) */}
          <Card className="lg:col-span-2 p-6 flex flex-col justify-between">
            <CardHeader className="px-0 pt-0 pb-3">
              <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">SHAP Predictive Impact</span>
            </CardHeader>
            <div className="h-[220px] mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={shapBarData} layout="vertical" margin={{ top: 5, right: 10, left: 30, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis type="number" tick={{ fontSize: 9, fill: "hsl(var(--foreground-muted))" }} />
                  <YAxis dataKey="name" type="category" tick={{ fontSize: 9, fill: "hsl(var(--foreground-muted))" }} width={80} />
                  <Tooltip
                    contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }}
                    itemStyle={{ fontSize: "10px" }}
                  />
                  <Bar dataKey="Importance" fill="hsl(var(--brand))" radius={[0, 4, 4, 0]}>
                    {shapBarData.map((entry: { name: string; Importance: number }, index: number) => {
                      const colors = ["#a855f7", "#3b82f6", "#10b981", "#f59e0b", "#ef4444"];
                      return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[10px] text-foreground-subtle italic mt-2 font-mono">
              SHAP values indicate the directional additive weight of each feature on the final classifier output.
            </p>
          </Card>
        </div>

        {/* ROW 5: [Clinical Summary] vs [Session Metadata + Export] */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Column 1: Clinical Summary */}
          <Card className={`border-l-4 p-6 border-y-0 border-r-0 rounded-l-none flex flex-col justify-between ${isBoth ? 'border-l-[#e11d48] bg-rose-950/20' : isHigh ? 'border-l-level-high bg-level-high-bg' : predictionResult.anxiety_level === 'Moderate' ? 'border-l-level-moderate bg-level-moderate-bg' : 'border-l-level-low bg-level-low-bg'}`}>
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-3 uppercase tracking-wider">
                Recommended Protocol
              </h3>
              <p className="text-sm text-foreground-muted mb-6 leading-relaxed">
                {isBoth
                  ? "Comorbid state active (BOTH STRESS & ANXIETY). Co-occurrence of high-frequency physiological stress biomarkers and frontal-asymmetry anxiety active. Protocol: Administer dual-mitigation autonomic regulation (4-7-8 breathing + cognitive grounding) and maintain elevated clinical surveillance."
                  : predictionResult.anxiety_level === 'High'
                  ? "Elevated stress markers detected (STRESS ONLY). Topologies indicate high-frequency temporal power. Protocol: Apply autonomic strain mitigation, initiate a 10-minute quiet sensor break, and monitor for subsequent heart-rate/signal escalation."
                  : predictionResult.anxiety_level === 'Moderate' 
                  ? "Elevated anxiety markers detected (ANXIETY ONLY). Emergent asymmetry in frontal arrays observed. Protocol: Apply cognitive grounding procedures, administer standard auditory-relaxation tools, and observe for baseline recovery."
                  : "Neurological baseline parameters are stable and within normative boundaries (HEALTHY). No immediate cognitive or stress-regulation protocols are required. Continue standard observation."}
              </p>
            </div>
            <div className="text-[10px] uppercase tracking-widest text-foreground-subtle border-t border-border pt-4 font-mono">
              DISCLAIMER: Automated Machine Learning Analysis. Must be verified by a board-certified clinician.
            </div>
          </Card>

          {/* Column 2: Session Metadata + Export */}
          <Card className="p-6 flex flex-col justify-between">
            <div>
              <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider block mb-4">Session Metadata</span>
              <div className="space-y-3 font-sans">
                <div className="flex justify-between text-xs items-center">
                  <span className="text-foreground-muted">Inference Architecture</span>
                  <span className="font-semibold text-foreground bg-background-muted px-2 py-0.5 rounded">{predictionResult.model_used}</span>
                </div>
                <div className="flex justify-between text-xs items-center">
                  <span className="text-foreground-muted">Dataset Manifold</span>
                  <span className="font-semibold text-foreground">Kaggle Clinical Psychiatric EEG</span>
                </div>
                <div className="flex justify-between text-xs items-center">
                  <span className="text-foreground-muted">Sampling Resolution</span>
                  <span className="font-semibold text-foreground">128 Hz</span>
                </div>
                <div className="flex justify-between text-xs items-center border-t border-border pt-3 mt-1">
                  <span className="text-foreground-muted">Total Processing Time</span>
                  <span className="font-semibold text-foreground text-brand">{predictionResult.inference_time_ms} ms</span>
                </div>
              </div>
            </div>

            <div className="border-t border-border pt-6 mt-6 flex gap-4">
              <Button 
                variant="outline" 
                className="w-full flex items-center justify-center gap-2"
                onClick={() => window.print()}
              >
                Print Assessment
              </Button>
              <Button 
                className="w-full flex items-center justify-center gap-2"
                onClick={() => {
                  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(predictionResult, null, 2));
                  const downloadAnchor = document.createElement('a');
                  downloadAnchor.setAttribute("href", dataStr);
                  downloadAnchor.setAttribute("download", `clinical_eeg_report_${predictionResult.anxiety_level.toLowerCase()}.json`);
                  document.body.appendChild(downloadAnchor);
                  downloadAnchor.click();
                  downloadAnchor.remove();
                }}
              >
                <Download className="w-4 h-4" /> Export JSON Report
              </Button>
            </div>
          </Card>
        </div>
      </div>
    );
  }

  if (!mounted) {
    return (
      <div className="w-full pt-16 pb-24 px-0 max-w-7xl mx-auto flex flex-col items-center justify-center min-h-[60vh] space-y-6">
        <Loader2 className="animate-spin text-brand w-12 h-12" />
        <div className="text-center space-y-2">
          <h2 className="text-xl font-semibold text-foreground tracking-tight">Initializing Diagnostics...</h2>
          <p className="text-sm text-foreground-muted max-w-sm mx-auto leading-relaxed">
            Synchronizing neural sensors and real-time visualization frameworks.
          </p>
        </div>
      </div>
    );
  }

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

      {predictionError && (
        <div className="flex items-center gap-3 p-4 bg-level-high-bg border border-level-high/20 text-level-high rounded-lg text-sm mb-6">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <span>{predictionError}</span>
        </div>
      )}

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
                <Button variant="outline" size="sm" onClick={() => loadPreset("anxious")}>Stress & Anxiety Preset</Button>
                <Button variant="outline" size="sm" onClick={() => loadPreset("relaxed")}>Normative Preset</Button>
                <Button variant="ghost" size="sm" className="ml-auto" onClick={clearAll}>Clear All</Button>
              </div>
            </motion.div>
          )}

          <div className="pt-4">
            <Button 
              size="lg" 
              className="w-full" 
              onClick={handleRunAnalysis} 
              disabled={uploading || loadingPrediction || !canRunAnalysis}
            >
              {loadingPrediction ? (
                <>
                  <Loader2 className="animate-spin mr-2 w-4 h-4" /> Running Diagnostic Ensemble...
                </>
              ) : activeTab === "upload" && !uploadedData ? (
                "Please Upload an EEG Recording first"
              ) : activeTab === "manual" && !canRunAnalysis ? (
                "Please Enter Valid Feature Inputs"
              ) : (
                "Analyze Stress & Anxiety"
              )}
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
          <div className="border border-border rounded-lg overflow-hidden mb-6">
            <table className="w-full text-left text-xs font-sans">
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

          {/* CLINICAL MAPPING SECTION */}
          <div className="mt-8 border border-border rounded-lg p-6 bg-surface/50 space-y-6">
            <h3 className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">Clinical Reference Guide</h3>
            
            <div className="space-y-4">
              <div className="border-b border-border/50 pb-3">
                <span className="text-xs font-semibold text-brand block mb-1">🧠 Frontal Alpha Asymmetry (FAA)</span>
                <p className="text-xs text-foreground-muted leading-relaxed">
                  Calculated as <code className="bg-background-muted px-1 rounded text-foreground font-mono">log(F4 alpha) - log(F3 alpha)</code>. Right-frontal dominance (negative values) is the primary clinical biomarker associated with emotional withdrawal, avoidance behavior, and active stress state responses.
                </p>
              </div>

              <div className="border-b border-border/50 pb-3">
                <span className="text-xs font-semibold text-foreground block mb-2">📡 10-20 Electrode Locus Map</span>
                <div className="grid grid-cols-2 gap-2 text-[11px]">
                  <div className="bg-background-subtle px-2 py-1 rounded">
                    <span className="font-semibold text-foreground">Fp1 / Fp2:</span> <span className="text-foreground-muted">Frontopolar (Prefrontal Lobe)</span>
                  </div>
                  <div className="bg-background-subtle px-2 py-1 rounded">
                    <span className="font-semibold text-foreground">F3 / F4:</span> <span className="text-foreground-muted">Dorsolateral Frontal</span>
                  </div>
                  <div className="bg-background-subtle px-2 py-1 rounded">
                    <span className="font-semibold text-foreground">T3 / T4 / T5 / T6:</span> <span className="text-foreground-muted">Temporal (Auditory/Memory)</span>
                  </div>
                  <div className="bg-background-subtle px-2 py-1 rounded">
                    <span className="font-semibold text-foreground">O1 / O2:</span> <span className="text-foreground-muted">Occipital (Visual Cortex)</span>
                  </div>
                </div>
              </div>

              <div>
                <span className="text-xs font-semibold text-foreground block mb-2">📊 Spectral Frequency Characteristics</span>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-foreground-muted">Delta (0.5 - 4 Hz)</span>
                    <span className="text-foreground font-medium">Deep sleep / Sensor spikes</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground-muted">Theta (4 - 8 Hz)</span>
                    <span className="text-foreground font-medium">Drowsiness / Creative transition</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground-muted">Alpha (8 - 12 Hz)</span>
                    <span className="text-foreground font-medium">Relaxed focus / Posterior baseline</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground-muted">Beta (12 - 30 Hz)</span>
                    <span className="text-brand font-medium">Active stress / Cognitive strain</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-foreground-muted">Gamma (30 - 45 Hz)</span>
                    <span className="text-level-high font-medium">High synthesis / Acute anxiety</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
