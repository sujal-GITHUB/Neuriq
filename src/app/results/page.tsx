"use client"

import * as React from "react"
import { Card, CardHeader, CardContent } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { DataTable } from "@/components/ui/DataTable"
import { Progress } from "@/components/ui/Progress"
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, 
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip
} from "recharts"

export default function ResultsPage() {
  const [data, setData] = React.useState<any>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    // We send a mock request to the backend. The python ML service 
    // mock_responses.py will return a randomized realistic DASPS JSON.
    const reqBody = {
      signals: [[0.1, 0.2], [0.3, 0.4]],
      channels: ["F3", "F4", "AF3", "F7"],
      sampling_rate: 128,
      modalities: ["EEG"],
      model: "boosting_ensemble"
    };

    fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody)
    })
      .then(res => res.json())
      .then(result => {
        setData(result);
        setLoading(false);
      })
      .catch(err => {
        console.error("Backend error:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="w-full h-[60vh] flex flex-col items-center justify-center space-y-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-brand"></div>
        <p className="text-foreground-muted text-sm tracking-widest uppercase">Processing ML Pipeline...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="w-full h-[60vh] flex flex-col items-center justify-center space-y-4">
        <p className="text-level-high font-medium">Failed to connect to ML Backend</p>
        <p className="text-sm text-foreground-muted">Make sure the Python API is running on port 8000.</p>
      </div>
    );
  }

  const isHigh = data.anxiety_level === "High";
  const confidencePercent = (data.confidence * 100).toFixed(1);

  const parsedBandData = Object.entries(data.band_powers).map(([k, v]) => ({
    subject: k.charAt(0).toUpperCase() + k.slice(1),
    A: v,
    fullMark: 50
  }));

  const parsedFeaturesData = data.top_features.slice(0, 5).map((f: any, index: number) => ({
    id: index + 1,
    feature: f.name,
    channel: f.name.split('_').pop(),
    band: f.name.includes("Alpha") ? "Alpha" : f.name.includes("Beta") ? "Beta" : f.name.includes("Theta") ? "Theta" : f.name.includes("Gamma") ? "Gamma" : "Complex",
    shap: f.importance,
    dir: f.value > 10 ? "up" : "down"
  }));

  return (
    <div className="w-full max-w-7xl mx-auto pt-16 pb-24 px-0 space-y-8">
      
      {/* HEADER */}
      <div className="border-b border-border pb-8 flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <div className="text-[10px] text-foreground-subtle mb-2 uppercase tracking-widest font-semibold">Dashboard / Analysis Report</div>
          <h1 className="text-3xl font-semibold text-foreground tracking-tight">Clinical Neurological Evaluation</h1>
          <p className="text-sm text-foreground-muted mt-2 max-w-xl leading-relaxed">
            Inference derived from uploaded EEG sequence. Analysis maps non-linear topologies directly to the psychological stimuli markers from the DASPS baseline dataset.
          </p>
        </div>
        <div className="text-left md:text-right bg-surface border border-border p-3 rounded-md">
           <div className="text-[10px] text-foreground-subtle uppercase tracking-widest mb-1">Generated Evaluation ID</div>
           <div className="font-mono text-sm text-foreground font-medium">AX-{Math.floor(Math.random() * 90000) + 10000}-DASPS</div>
        </div>
      </div>

      {/* TOP ROW */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card variant="metric" className="p-6 relative overflow-hidden">
          <div className={`absolute top-0 right-0 w-2 h-full ${isHigh ? 'bg-level-high' : data.anxiety_level === 'Moderate' ? 'bg-level-moderate' : 'bg-level-low'}`} />
          <div className="flex justify-between items-start mb-4">
            <span className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Detection Status</span>
            <Badge variant={isHigh ? "high" : data.anxiety_level === 'Moderate' ? "outline" : "low"}>
              {isHigh ? "Action Required" : data.anxiety_level === 'Moderate' ? "Monitor" : "Normative"}
            </Badge>
          </div>
          <div className={`text-3xl font-semibold mb-1 ${isHigh ? 'text-level-high' : data.anxiety_level === 'Moderate' ? 'text-level-moderate' : 'text-level-low'}`}>
            {data.anxiety_level.toUpperCase()} ANXIETY
          </div>
          <div className="text-sm text-foreground-muted">Model Confidence: {confidencePercent}%</div>
        </Card>

        <Card className="p-6">
          <span className="text-xs font-medium text-foreground-muted uppercase tracking-wider block mb-4">Softmax Class Probabilities</span>
          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-foreground-muted">High Cognitive Strain</span>
                <span className="font-medium text-foreground tabular-nums">{(data.probabilities.High * 100).toFixed(1)}%</span>
              </div>
              <Progress value={data.probabilities.High * 100} indicatorClassName="bg-level-high" />
            </div>
            <div>
               <div className="flex justify-between text-xs mb-1">
                <span className="text-foreground-muted">Moderate / Baseline Transition</span>
                <span className="font-medium text-foreground tabular-nums">{(data.probabilities.Moderate * 100).toFixed(1)}%</span>
              </div>
              <Progress value={data.probabilities.Moderate * 100} indicatorClassName="bg-level-moderate" />
            </div>
            <div>
               <div className="flex justify-between text-xs mb-1">
                <span className="text-foreground-muted">Low / Relaxed State</span>
                <span className="font-medium text-foreground tabular-nums">{(data.probabilities.Low * 100).toFixed(1)}%</span>
              </div>
              <Progress value={data.probabilities.Low * 100} indicatorClassName="bg-level-low" />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <span className="text-xs font-medium text-foreground-muted uppercase tracking-wider block mb-4">Inference Telemetry</span>
          <div className="space-y-3">
            <div className="flex justify-between text-xs items-center">
              <span className="text-foreground-muted">Inference Architecture</span>
              <span className="font-medium text-foreground bg-background-muted px-2 py-0.5 rounded">{data.model_used}</span>
            </div>
            <div className="flex justify-between text-xs items-center">
              <span className="text-foreground-muted">Dataset Manifold</span>
              <span className="font-medium text-foreground">DASPS (14-Ch EPOC+)</span>
            </div>
            <div className="flex justify-between text-xs items-center">
              <span className="text-foreground-muted">Sampling Resolution</span>
              <span className="font-medium text-foreground">128 Hz</span>
            </div>
             <div className="flex justify-between text-xs items-center border-t border-border pt-3 mt-1">
              <span className="text-foreground-muted">Total Processing Time</span>
              <span className="font-medium text-foreground text-brand">{data.inference_time_ms} ms</span>
            </div>
          </div>
        </Card>
      </div>

      {/* SECOND ROW */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <Card className="col-span-3">
          <CardHeader>
             <span className="text-xs font-medium text-foreground-muted uppercase tracking-wider">Topological Spectral Distribution</span>
          </CardHeader>
          <CardContent className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="70%" data={parsedBandData}>
                <PolarGrid stroke="hsl(var(--border))" strokeDasharray="3 3" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: "hsl(var(--foreground-muted))" }} />
                <Radar name="Power" dataKey="A" stroke="hsl(var(--brand))" strokeWidth={1.5} fill="transparent" />
                <Tooltip 
                  contentStyle={{ backgroundColor: "hsl(var(--surface))", borderColor: "hsl(var(--border))", borderRadius: "6px" }} 
                  itemStyle={{ color: "hsl(var(--foreground))", fontSize: "12px" }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="col-span-2 flex flex-col justify-center items-center text-center p-8 bg-gradient-to-br from-surface to-background transition-all">
           <span className="text-xs font-medium text-foreground-muted uppercase tracking-wider mb-6 w-full text-left">Frontal Alpha Asymmetry (FAA) Index</span>
           <div className={`text-5xl font-semibold tabular-nums mb-4 ${data.frontal_asymmetry < 0 ? 'text-level-high' : 'text-brand'}`}>
             {data.frontal_asymmetry > 0 ? "+" : ""}{data.frontal_asymmetry}
           </div>
           
           <div className="w-full relative py-2">
             <div className="w-full h-1.5 rounded-full overflow-hidden bg-gradient-to-r from-level-high via-level-moderate to-level-low" />
             {/* Indicator tick representing the asymmetry value between -0.5 and 0.5 roughly */}
             <div 
               className="absolute top-0 bottom-0 w-1 bg-foreground rounded shadow-sm" 
               style={{ left: `${Math.max(0, Math.min(100, (data.frontal_asymmetry + 0.5) * 100))}%` }} 
             />
           </div>

           <div className="flex w-full justify-between text-[10px] text-foreground-subtle mt-1 uppercase tracking-widest mb-6 font-semibold">
             <span>Right-Dom (Avoidance)</span>
             <span>Left-Dom (Approach)</span>
           </div>
           <p className="text-xs text-foreground-muted text-balance leading-relaxed">
             Highly negative values indicate right-frontal alpha dominance. In the context of the DASPS elicitation paradigm, this correlates strongly with negative affect, psychological anxiety, and active emotional withdrawal responses.
           </p>
        </Card>
      </div>

      {/* THIRD ROW - FEATURES */}
      <div>
        <h2 className="text-lg font-semibold text-foreground mb-1">Global Feature Attributions</h2>
        <p className="text-sm text-foreground-muted mb-4">Primary neurological predictors driving this specific inference pipeline, ordered by SHAP (SHapley Additive exPlanations) discriminative impact.</p>
        
        <div className="border border-border rounded-lg overflow-hidden bg-surface">
          <DataTable
            data={parsedFeaturesData}
            columns={[
              { key: "id", header: "Rank", render: (r) => <span className="text-foreground-subtle font-mono text-xs">0{r.id}</span> },
              { key: "feature", header: "Biomarker Metric", sortable: true, render: (r) => <span className="font-medium text-foreground">{r.feature.replace(/_/g, ' ')}</span> },
              { key: "channel", header: "Electrode Site", render: (r) => <Badge variant="outline" className="font-mono">{r.channel}</Badge> },
              { key: "band", header: "Spectral Core" },
              { key: "shap", header: "SHAP Impact", sortable: true, render: (r) => <span className="font-mono text-xs text-brand">{r.shap.toFixed(4)}</span> },
              { key: "dir", header: "Vector", render: (r) => (
                <Badge variant={r.dir === "up" ? "high" : "low"}>{r.dir === "up" ? "↑ Excitatory" : "↓ Inhibitory"}</Badge>
              )}
            ]}
          />
        </div>
      </div>

      {/* RECOMMENDATIONS */}
      <h2 className="text-lg font-semibold text-foreground mt-8 border-t border-border pt-8">Clinical Summary & Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
        <Card className={`border-l-4 p-6 border-y-0 border-r-0 rounded-l-none ${isHigh ? 'border-l-level-high bg-level-high-bg' : data.anxiety_level === 'Moderate' ? 'border-l-level-moderate bg-level-moderate-bg' : 'border-l-level-low bg-level-low-bg'}`}>
          <h3 className="text-sm font-semibold text-foreground mb-3 uppercase tracking-wider">
            Recommended Protocol
          </h3>
          <p className="text-sm text-foreground-muted mb-4 leading-relaxed">
            {isHigh 
              ? "Elevated neurological anxiety markers detected. Topology mirrors DASPS anxiety-elicitation profiles. Protocol: Initiate immediate reduction of psychological triggers, apply clinical grounding techniques, and administer autonomic-regulation exercises. Mandate 15-minute observational re-evaluation."
              : data.anxiety_level === 'Moderate' 
              ? "Moderate arousal detected. Topologies indicate active processing or mild strain. Monitor for escalation and maintain normative baseline."
              : "Neurological baseline parameters are stable and remain within normative boundaries. No immediate cognitive regulation protocol is required. Continue standard observation."}
          </p>
          <div className="text-[10px] uppercase tracking-widest text-foreground-subtle border-t border-border pt-4">
            DISCLAIMER: Automated Machine Learning Analysis. Must be verified by a board-certified clinician.
          </div>
        </Card>

        <Card className="p-6 bg-surface border-border">
          <h3 className="text-sm font-semibold text-foreground mb-3 uppercase tracking-wider">
            Neurological Context
          </h3>
          <p className="text-sm text-foreground-muted leading-relaxed">
            Inference engine leverages the explicit subjective-reporting matrix from the DASPS psychological stimuli model. Evaluations lean heavily on frontal asymmetry metrics (F3/F4 array locus) and emergent high-frequency Beta/Gamma bands in temporal sites (T7/T8), known biomarkers of acute hyperarousal and elevated emotional regulation effort.
          </p>
        </Card>
      </div>

    </div>
  )
}
