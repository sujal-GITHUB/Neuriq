import Link from 'next/link'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { Brain, Activity, LineChart } from 'lucide-react'

export default function Home() {
  return (
    <div className="w-full">
      {/* HERO SECTION */}
      <section className="py-32 px-6 max-w-3xl mx-auto text-center">
        <div className="inline-flex items-center gap-1.5 text-xs border border-border rounded-full px-3 py-1 text-foreground-muted mb-8">
          DASPS Dataset &mdash; 23 Subjects &mdash; 14-Channel EEG
        </div>
        <h1 className="text-5xl font-semibold tracking-tight leading-tight text-foreground max-w-2xl mx-auto">
          Clinical Anxiety Detection from EEG Brain Signals
        </h1>
        <p className="text-lg text-foreground-muted mt-4 max-w-xl mx-auto leading-relaxed font-normal">
          Analyze EEG signals using advanced ML and DL models to classify mental anxiety levels. 
          Built on peer-reviewed research for real-time monitoring.
        </p>
        <div className="mt-10 flex gap-3 justify-center">
          <Link href="/analyze">
            <Button size="lg">Start Analysis</Button>
          </Link>
          <Link href="/research">
            <Button variant="outline" size="lg">View Research</Button>
          </Link>
        </div>
        <p className="text-xs text-foreground-subtle mt-6">
          Based on Baghdadi et al. (2019) and Aldayel &amp; Al-Nafjan (2024)
        </p>
      </section>

      {/* METRICS STRIP */}
      <section className="border-y border-border py-8 mt-24">
        <div className="max-w-4xl mx-auto px-6 grid grid-cols-1 sm:grid-cols-4 gap-8">
          <div className="text-center">
            <div className="text-3xl font-semibold text-foreground tabular-nums">23</div>
            <div className="text-xs text-foreground-muted mt-1 uppercase tracking-wider">Subjects</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-semibold text-foreground tabular-nums">14</div>
            <div className="text-xs text-foreground-muted mt-1 uppercase tracking-wider">EEG Channels</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-semibold text-foreground tabular-nums">87.5%</div>
            <div className="text-xs text-foreground-muted mt-1 uppercase tracking-wider">Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-semibold text-foreground tabular-nums">128</div>
            <div className="text-xs text-foreground-muted mt-1 uppercase tracking-wider">Hz</div>
          </div>
        </div>
      </section>

      {/* FEATURES SECTION */}
      <section className="py-24 max-w-7xl mx-auto px-6">
        <div className="text-xs font-medium text-brand uppercase tracking-widest mb-3">Capabilities</div>
        <h2 className="text-2xl font-semibold text-foreground">Advanced Analysis Features</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 mt-12 gap-8">
          <div className="border border-border rounded-lg p-6">
            <div className="w-8 h-8 rounded-md bg-brand/10 flex items-center justify-center mb-4 text-brand">
              <Brain className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-medium text-foreground">Deep Learning Models</h3>
            <p className="text-sm text-foreground-muted mt-1 leading-relaxed">
              Utilizing CNN-BiLSTM and Brain2Vec architectures to extract spatio-temporal dynamics from raw EEG.
            </p>
          </div>
          <div className="border border-border rounded-lg p-6">
            <div className="w-8 h-8 rounded-md bg-brand/10 flex items-center justify-center mb-4 text-brand">
              <Activity className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-medium text-foreground">Multi-Domain Extraction</h3>
            <p className="text-sm text-foreground-muted mt-1 leading-relaxed">
              Processing inputs across time, frequency, and non-linear domains for highly robust feature matrices.
            </p>
          </div>
          <div className="border border-border rounded-lg p-6">
            <div className="w-8 h-8 rounded-md bg-brand/10 flex items-center justify-center mb-4 text-brand">
              <LineChart className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-medium text-foreground">Clinical Explainability</h3>
            <p className="text-sm text-foreground-muted mt-1 leading-relaxed">
              Extracting frontal asymmetry indices and SHAP values to explain neurological predictions.
            </p>
          </div>
        </div>
      </section>

      {/* PIPELINE SECTION */}
      <section className="py-24 bg-background-subtle border-y border-border">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-2xl font-semibold text-foreground mb-12">Methodology Pipeline</h2>
          
          <div className="flex flex-col md:flex-row justify-between relative">
            <div className="hidden md:block absolute top-[15px] left-0 w-full h-px bg-border -z-0" />
            
            {["EEG Recording", "Preprocessing", "Feature Extraction", "Classification", "Results"].map((step, i) => (
              <div key={step} className="flex flex-col items-center relative z-10 bg-background-subtle px-2 mb-8 md:mb-0">
                <div className="w-8 h-8 rounded-full border-2 border-brand bg-background flex items-center justify-center text-xs font-semibold text-foreground mb-3">
                  {i + 1}
                </div>
                <div className="text-sm font-medium text-foreground">{step}</div>
              </div>
            ))}
          </div>
          
          <div className="mt-12 text-center max-w-2xl mx-auto">
            <p className="text-sm text-foreground-muted leading-relaxed">
              Signals are recorded at 128Hz via 14 channels before being subjected to strict bandpass filtering. 
              Subsequent phases map extracted temporal structures securely into our classification endpoint to determine live anxiety levels with minimum latency.
            </p>
          </div>
        </div>
      </section>

      {/* RESEARCH CITATIONS */}
      <section className="py-24 max-w-7xl mx-auto px-6">
        <h2 className="text-2xl font-semibold text-foreground mb-12">Referenced Research</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="border border-border rounded-lg p-6">
            <Badge variant="outline">Peer Reviewed</Badge>
            <h3 className="text-sm font-medium text-foreground mt-3">
              DASPS: A database for anxious states based on a psychological stimulation
            </h3>
            <p className="text-xs text-foreground-subtle mt-1">Baghdadi et al., 2019</p>
            <p className="text-xs text-foreground-muted mt-3">
              Validated physiological changes specifically during cognitively challenging logical assessments under tight constraints.
            </p>
          </div>
          <div className="border border-border rounded-lg p-6">
            <Badge variant="outline">Peer Reviewed</Badge>
            <h3 className="text-sm font-medium text-foreground mt-3">
              Machine Learning for Cognitive Load Detection in Working Environment
            </h3>
            <p className="text-xs text-foreground-subtle mt-1">Aldayel &amp; Al-Nafjan, 2024</p>
            <p className="text-xs text-foreground-muted mt-3">
              Established 87.5% classification accuracy on DASPS utilizing Gradient Bagging Ensembles evaluated via 5-Fold Cross Validation.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}
