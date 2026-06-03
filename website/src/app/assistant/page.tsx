"use client"

import * as React from "react"
import { Button } from "@/components/ui/Button"
import { Card, CardHeader, CardContent } from "@/components/ui/Card"
import { Input } from "@/components/ui/Input"
import {
  Send, Brain, HelpCircle, ChevronRight, RefreshCw, BookOpen
} from "lucide-react"

// RAG Clinical Database mapped to DEAP Valence-Arousal Coordinates
const RAG_DATABASE = [
  {
    id: "low_val_high_arous",
    title: "Low Valence / High Arousal (Acute Stress & Panic)",
    valence: "< 5.0",
    arousal: "> 5.0",
    inferred_state: "Acute Stress / Distress / Panic",
    keywords: ["panic", "distress", "acute stress", "high arousal", "low valence", "overstimulated", "fight or flight", "anxiety spike", "anxious"],
    eeg_characteristics: "Massive suppression of Alpha band power across all channels, highly elevated high-beta and gamma power, pronounced negative Frontal Alpha Asymmetry (high right-hemisphere activity), elevated delta baseline.",
    description: "Autonomic down-regulation target. Critical levels of brain wave hyperactivity showing acute emotional distress and profound sympathetic dominance.",
    remedy_focus: "Parasympathetic activation, immediate down-regulation, and acute panic management.",
    remedies: {
      bio: "4-7-8 deep diaphragmatic breathing or physiological sighs (two quick inhales through the nose, one long exhale through the mouth).",
      sensory: "Closed-loop binaural beats set specifically to low-alpha (8–10 Hz) or theta (4–7 Hz) to suppress active beta spikes.",
      behavioral: "Immediate cold exposure (splash cold water on face to trigger mammalian dive reflex to drop heart rate) or progressive muscle relaxation (PMR)."
    },
    metadata: {
      valence_tier: "low",
      arousal_tier: "high",
      remedy_type: "somatic_calming"
    }
  },
  {
    id: "low_val_low_arous",
    title: "Low Valence / Low Arousal (Cognitive Fatigue & Burnout)",
    valence: "< 5.0",
    arousal: "< 5.0",
    inferred_state: "Cognitive Fatigue / Depression / Burnout",
    keywords: ["fatigue", "burnout", "exhausted", "depression", "sluggish", "lethargic", "flat", "low arousal", "low valence"],
    eeg_characteristics: "Sluggish alpha activity, heavy delta/theta waves during active tasks, suppressed high-beta frequency power in prefrontal nodes.",
    description: "Neural activation and mood-lifting target. Reduced cognitive pacing and neuro-chemical exhaustion.",
    remedy_focus: "Neural up-regulation, dopaminergic stimulation, and cognitive pacing.",
    remedies: {
      bio: "High-intensity somatic breathing (e.g., Wim Hof style breathwork) to safely drive up physiological arousal.",
      sensory: "Binaural beats or ambient audio utilizing gamma/high-beta sweeps to encourage sharp cognitive focus and neural activation.",
      behavioral: "Behavioral activation therapy scripts, 10-minute brisk aerobic walks, or targeted bright light therapy to reset natural circadian dopamine levels."
    },
    metadata: {
      valence_tier: "low",
      arousal_tier: "low",
      remedy_type: "neuro_stimulating"
    }
  },
  {
    id: "high_val_high_arous",
    title: "High Valence / High Arousal (Excitement & Flow)",
    valence: "> 5.0",
    arousal: "> 5.0",
    inferred_state: "Excitement / Hyperactivity",
    keywords: ["excitement", "hyperactivity", "flow", "high arousal", "high valence", "energized", "focused"],
    eeg_characteristics: "Elevated frontal theta paired with high-frequency relative beta power, optimal sensory motor rhythm (SMR).",
    description: "Focus alignment and grounded stimulation target. High emotional valence and positive sympathetic activation.",
    remedy_focus: "Focus alignment, grounded meditation, and sustained attention.",
    remedies: {
      bio: "Coherence breathing (equal 5s inhale, 5s exhale cycles) to balance autonomic activation.",
      sensory: "Mid-range alpha (10 Hz) binaural beats for flowing concentration and flow state maintenance.",
      behavioral: "Grounded body scans, structured priority planning, or focus journaling to channel high energy productively."
    },
    metadata: {
      valence_tier: "high",
      arousal_tier: "high",
      remedy_type: "flow_alignment"
    }
  },
  {
    id: "high_val_low_arous",
    title: "High Valence / Low Arousal (Calm, Rest & Recovery)",
    valence: "> 5.0",
    arousal: "< 5.0",
    inferred_state: "Calm / Rest / Recovery (Optimal Baseline)",
    keywords: ["calm", "rest", "recovery", "baseline", "relax", "meditative", "sleepy", "peaceful"],
    eeg_characteristics: "Strong relative alpha power in posterior/occipital nodes, positive frontal alpha asymmetry index (> 0), normative sample entropy, standard spectral complexity.",
    description: "Restorative neural plasticity target. Healthy baseline brain wave activity showing cognitive balance and optimal recovery.",
    remedy_focus: "State maintenance and restorative neural plasticity.",
    remedies: {
      bio: "Standard diaphragmatic breathing (4s inhale, 6s exhale) to maintain vagal tone.",
      sensory: "Delta sweeps (1–4 Hz) or deep alpha ambient soundscapes to promote deep rest.",
      behavioral: "Compassion mindfulness, light active stretches, or restorative gratitude journaling."
    },
    metadata: {
      valence_tier: "high",
      arousal_tier: "low",
      remedy_type: "restorative"
    }
  }
];

export default function AssistantPage() {
  const [input, setInput] = React.useState("")
  const [messages, setMessages] = React.useState<any[]>([
    {
      role: "assistant",
      content: "Hello! I am your clinical RAG-based EEG assistant. Ask me anything about your Valence-Arousal states, fatigue, acute panic, excitement, or recovery, and I will retrieve matching DEAP quadrant remedies.",
      retrievedDocs: []
    }
  ])
  const [loading, setLoading] = React.useState(false)
  const [retrievedDocs, setRetrievedDocs] = React.useState<any[]>([])
  const [activeScenario, setActiveScenario] = React.useState<string | null>(null)
  const messagesEndRef = React.useRef<HTMLDivElement>(null)

  // Auto scroll to bottom
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // RAG Retriever Engine mapped to Valence-Arousal quadrants
  const retrieveRelevantDocs = (query: string) => {
    const cleanQuery = query.toLowerCase()
    const scoredDocs = RAG_DATABASE.map(doc => {
      let score = 0
      doc.keywords.forEach(keyword => {
        if (cleanQuery.includes(keyword)) {
          score += 2
        }
      })
      if (cleanQuery.includes("valence") && cleanQuery.includes("arousal")) {
        score += 1
      }
      return { ...doc, score }
    })

    const matches = scoredDocs.filter(d => d.score > 0).sort((a, b) => b.score - a.score)
    if (matches.length === 0) {
      return []
    }
    return matches.slice(0, 2)
  }

  const handleSendMessage = async (textToSend?: string) => {
    const queryText = textToSend || input
    if (!queryText.trim()) return

    // 1. RAG retrieval step
    const relevantDocs = retrieveRelevantDocs(queryText)
    setRetrievedDocs(relevantDocs)

    // User message
    const userMsg = { role: "user", content: queryText }
    setMessages(prev => [...prev, userMsg])
    setInput("")
    setLoading(true)

    // 2. Build clinical system prompt from retrieved data
    let ragContext = "No specific retrieved EEG documents for this query."
    if (relevantDocs.length > 0) {
      ragContext = relevantDocs.map(doc => `
Document Title: ${doc.title}
Valence Target: ${doc.valence}
Arousal Target: ${doc.arousal}
Inferred Mental State: ${doc.inferred_state}
Focus Area: ${doc.remedy_focus}
EEG Characteristics: ${doc.eeg_characteristics}
Actionable RAG Prompt Output remedies:
- Bio-remedy: ${doc.remedies.bio}
- Sensory-remedy: ${doc.remedies.sensory}
- Behavioral: ${doc.remedies.behavioral}
`).join("\n---\n")
    }

    const systemPrompt = `
You are a research-grade clinical EEG psychiatrist assistant inside Neuriq.
Your task is to answer the patient's questions strictly using the retrieved EEG DEAP Valence-Arousal RAG context provided below.

=== RETRIEVED EEG RAG CONTEXT ===
${ragContext}
================================

Guidelines & System Prompt Tuning:
1. Ground your answers in the clinical EEG coordinates, characteristics, and remedies provided.
2. Formulate your response with an estimated "Classification Confidence Score" (e.g. "Classification Confidence Score: 94.5%") explicitly displayed at the very top.
3. Follow the confidence score with direct, step-by-step instructions.
4. Structure the remedies clearly into exactly these three distinct sections:
   - **Bio-remedy**: [Summarize the biological/breathing remedy]
   - **Sensory-remedy**: [Summarize the sound/music sweeps/binaural frequencies]
   - **Behavioral**: [Summarize somatic, cold exposure, or physical exercises]
5. Always preface your reply with a gentle research-grade advisory stating that Neuriq is a support tool and clinical consultations are recommended for formal diagnoses.
`

    try {
      const response = await fetch("/api/assistant", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          messages: messages.map(m => ({ role: m.role, content: m.content })),
          systemPrompt: systemPrompt
        })
      })

      if (!response.ok) {
        const errText = await response.json();
        throw new Error(errText.error || `Failed with status ${response.status}`)
      }

      const resData = await response.json()
      const assistantContent = resData.content || "No response received from AI."

      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: assistantContent,
          retrievedDocs: relevantDocs
        }
      ])
    } catch (err: any) {
      console.error(err)
      setMessages(prev => [
        ...prev,
        {
          role: "assistant",
          content: `❌ Error: ${err.message || "Unknown error."}`
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex w-full min-h-[calc(100vh-3.5rem)] pt-14 -mt-14 relative bg-background">

      {/* MAIN WORKSPACE CONTENT */}
      <main className="flex-1 py-8 px-4 max-w-7xl mx-auto">
        <div className="mb-8 border-b border-border pb-8">
          <div className="text-xs text-foreground-subtle mb-2">Dashboard / RAG Assistant</div>
          <h1 className="text-3xl font-semibold text-foreground tracking-tight flex items-center gap-2">
            <Brain className="h-7 w-7 text-brand animate-pulse" /> Clinical RAG AI Assistant
          </h1>
          <p className="text-sm text-foreground-muted mt-1 max-w-xl">
            Retrieve DEAP-filtered autonomic remedies, sensory setups, and physiological sighs based on real-time valence-arousal EEG coordinates.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-stretch">

          {/* Chat Interface Container */}
          <div className="lg:col-span-2 flex flex-col border border-border rounded-lg bg-surface overflow-hidden min-h-[500px]">

            {/* Chat Logs */}
            <div className="flex-1 p-5 overflow-y-auto max-h-[420px] space-y-4 bg-background-subtle/20">
              {messages.map((msg, index) => {
                const isAssistant = msg.role === "assistant"
                return (
                  <div key={index} className={`flex gap-3 ${isAssistant ? "justify-start" : "justify-end"}`}>
                    {isAssistant && (
                      <div className="w-8 h-8 rounded-lg bg-brand/10 border border-brand/20 flex items-center justify-center text-brand flex-shrink-0 mt-0.5">
                        <Brain className="h-4 w-4" />
                      </div>
                    )}
                    <div className={`max-w-[85%] rounded-lg p-3 text-sm leading-relaxed border ${isAssistant
                      ? "bg-background-subtle border-border text-foreground"
                      : "bg-brand/10 border-brand/20 text-foreground ml-auto"
                      }`}>
                      <p className="whitespace-pre-wrap">{msg.content}</p>

                      {isAssistant && msg.retrievedDocs && msg.retrievedDocs.length > 0 && (
                        <div className="mt-3 pt-2.5 border-t border-border/50 flex flex-wrap gap-1.5 items-center">
                          <span className="text-[10px] uppercase font-bold tracking-wider text-foreground-muted flex items-center gap-1">
                            <BookOpen className="h-3 w-3 text-brand" /> RAG References:
                          </span>
                          {msg.retrievedDocs.map((doc: any) => (
                            <span key={doc.id} className="text-[10px] px-2 py-0.5 rounded bg-brand/5 border border-brand/20 text-brand font-medium">
                              {doc.title}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
              {loading && (
                <div className="flex gap-3 justify-start">
                  <div className="w-8 h-8 rounded-lg bg-brand/10 border border-brand/20 flex items-center justify-center text-brand flex-shrink-0">
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  </div>
                  <div className="max-w-[80%] rounded-lg p-3 text-sm bg-background-subtle border border-border text-foreground flex items-center gap-2">
                    <span>AI is querying the clinical EEG database and thinking...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Chat Inputs */}
            <div className="p-4 border-t border-border bg-background-subtle flex items-center gap-2">
              <Input
                placeholder="Ask about Valence-Arousal, acute stress, burnout remedies, or excitement..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSendMessage()
                }}
                disabled={loading}
                className="flex-1 bg-background border-border text-sm"
              />
              <Button size="sm" onClick={() => handleSendMessage()} disabled={loading || !input.trim()} className="bg-brand hover:bg-brand-hover">
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* RAG Retrieved Live Retrieval details column */}
          <div className="flex flex-col gap-4">
            <Card className="border border-border bg-surface flex-1 overflow-hidden relative flex flex-col">
              <CardHeader className="p-4 pb-0 flex flex-row items-center justify-between border-b border-border/50 pb-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-foreground-muted flex items-center gap-2">
                  <BookOpen className="h-4 w-4 text-brand" /> Live RAG Retrieval
                </h3>
                {retrievedDocs.length > 0 && (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-brand/10 border border-brand/20 text-brand font-bold">
                    {retrievedDocs.length} Active
                  </span>
                )}
              </CardHeader>
              <CardContent className="p-4 flex-1 overflow-y-auto space-y-4">
                {retrievedDocs.length > 0 ? (
                  retrievedDocs.map((doc) => (
                    <div key={doc.id} className="p-3 rounded-lg border border-brand/20 bg-brand/5 relative flex flex-col gap-2">
                      <div className="absolute top-0 left-0 w-1 h-full bg-brand" />
                      <h4 className="text-xs font-bold text-brand">{doc.title}</h4>
                      <div className="grid grid-cols-2 gap-1.5 text-[9px] text-foreground-subtle border-b border-border/30 pb-1.5 font-medium">
                        <span>Valence: {doc.valence}</span>
                        <span>Arousal: {doc.arousal}</span>
                      </div>
                      <div>
                        <span className="text-[9px] uppercase font-bold text-foreground-subtle">EEG Biomarkers</span>
                        <p className="text-[10px] text-foreground leading-relaxed mt-0.5">{doc.eeg_characteristics}</p>
                      </div>
                      <div className="space-y-1 mt-1 border-t border-border/30 pt-1.5">
                        <span className="text-[9px] uppercase font-bold text-foreground-subtle">RAG Remedies</span>
                        <div className="text-[10px] text-foreground leading-relaxed space-y-1.5">
                          <p><strong className="text-brand">Bio:</strong> {doc.remedies.bio}</p>
                          <p><strong className="text-brand">Sensory:</strong> {doc.remedies.sensory}</p>
                          <p><strong className="text-brand">Behavioral:</strong> {doc.remedies.behavioral}</p>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-center p-6 text-foreground-subtle border border-dashed border-border rounded-lg bg-background-subtle/20 min-h-[300px]">
                    <HelpCircle className="h-8 w-8 text-foreground-subtle mb-2" />
                    <span className="text-xs font-semibold text-foreground">Waiting for Retrieval</span>
                    <p className="text-[10px] mt-1 text-foreground-muted max-w-[180px] leading-relaxed">
                      Type a query about DEAP coordinates (valence/arousal) or fatigue to pull biomarkers and somatic remedies.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

        </div>
      </main>

    </div>
  )
}
