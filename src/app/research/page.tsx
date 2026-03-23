'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';

const timeline = [
  { year: '2018', title: 'EEGNet Published', desc: 'Lawhern et al. introduce compact CNN for EEG-based BCIs', color: '#6366f1' },
  { year: '2020', title: 'DL for Emotion', desc: 'Deep learning approaches gain traction for EEG emotion recognition', color: '#06b6d4' },
  { year: '2023', title: 'CNN-LSTM Architectures', desc: 'Hybrid CNN-LSTM models with attention show state-of-the-art results', color: '#22c55e' },
  { year: '2024', title: 'Meta-classifier Ensemble', desc: 'Gandhi & Jaliya propose hybrid features with meta-model for anxiety detection', color: '#f59e0b' },
  { year: '2024', title: 'Comprehensive Reviews', desc: 'Chaudhari & Shrivastava, Badr et al. publish systematic reviews of EEG anxiety detection', color: '#ef4444' },
  { year: '2025', title: 'Brain2Vec', desc: 'Mynoddin et al. propose CNN+LSTM+Attention framework evaluated on DEAP', color: '#8b5cf6' },
];

const bands = [
  { name: 'Delta', range: '0.5–4 Hz', normal: 'Deep sleep, unconscious states', anxiety: 'May increase in severe fatigue/burnout', color: '#3b82f6' },
  { name: 'Theta', range: '4–8 Hz', normal: 'Drowsiness, light sleep, meditation', anxiety: 'Increases with cognitive overload', color: '#06b6d4' },
  { name: 'Alpha', range: '8–13 Hz', normal: 'Relaxed wakefulness, eyes closed', anxiety: 'DECREASES — primary anxiety marker', color: '#22c55e' },
  { name: 'Beta', range: '13–30 Hz', normal: 'Active thinking, concentration', anxiety: 'INCREASES — cortical arousal marker', color: '#f59e0b' },
  { name: 'Gamma', range: '30–45 Hz', normal: 'High-level cognition, binding', anxiety: 'Variable — may increase with anxiety', color: '#ef4444' },
];

const datasets = [
  { name: 'DEAP', subjects: 32, channels: 32, peripheral: 8, sr: 512, labels: 'Valence, Arousal (1-9)', modalities: 'EEG, GSR, ECG, EOG, EMG, Temp, Resp', url: 'https://www.eecs.qmul.ac.uk/mmv/datasets/deap/', ref: 'Koelstra et al., 2012' },
  { name: 'DREAMER', subjects: 23, channels: 14, peripheral: 2, sr: 128, labels: 'Valence, Arousal (1-5)', modalities: 'EEG, ECG', url: 'https://zenodo.org/record/546113', ref: 'Katsigiannis & Ramzan, 2018' },
  { name: 'MAHNOB-HCI', subjects: 30, channels: 32, peripheral: 6, sr: 256, labels: 'Valence, Arousal (1-9)', modalities: 'EEG, GSR, ECG, Temp, Resp, Video, Gaze', url: 'https://mahnob-db.eu/hci-tagging/', ref: 'Soleymani et al., 2012' },
];

const electrodes: { id: string; cx: number; cy: number; label: string; desc: string }[] = [
  { id: 'Fp1', cx: 38, cy: 18, label: 'Fp1', desc: 'Left prefrontal — executive function, decision making' },
  { id: 'Fp2', cx: 62, cy: 18, label: 'Fp2', desc: 'Right prefrontal — emotional processing' },
  { id: 'F7', cx: 20, cy: 32, label: 'F7', desc: 'Left frontal — verbal working memory' },
  { id: 'F3', cx: 38, cy: 32, label: 'F3', desc: 'Left frontal — approach motivation (anxiety ↔ F4 asymmetry)' },
  { id: 'Fz', cx: 50, cy: 30, label: 'Fz', desc: 'Midline frontal — attention, error monitoring' },
  { id: 'F4', cx: 62, cy: 32, label: 'F4', desc: 'Right frontal — withdrawal motivation' },
  { id: 'F8', cx: 80, cy: 32, label: 'F8', desc: 'Right frontal — emotional arousal' },
  { id: 'T3', cx: 12, cy: 50, label: 'T3', desc: 'Left temporal — auditory, language' },
  { id: 'C3', cx: 35, cy: 50, label: 'C3', desc: 'Left central — motor cortex' },
  { id: 'Cz', cx: 50, cy: 48, label: 'Cz', desc: 'Midline central — sensorimotor integration' },
  { id: 'C4', cx: 65, cy: 50, label: 'C4', desc: 'Right central — motor cortex' },
  { id: 'T4', cx: 88, cy: 50, label: 'T4', desc: 'Right temporal — face/emotion recognition' },
  { id: 'P3', cx: 38, cy: 68, label: 'P3', desc: 'Left parietal — spatial awareness' },
  { id: 'Pz', cx: 50, cy: 66, label: 'Pz', desc: 'Midline parietal — P300 attention component' },
  { id: 'P4', cx: 62, cy: 68, label: 'P4', desc: 'Right parietal — visuospatial' },
  { id: 'O1', cx: 42, cy: 82, label: 'O1', desc: 'Left occipital — visual processing' },
  { id: 'O2', cx: 58, cy: 82, label: 'O2', desc: 'Right occipital — visual processing' },
];

const glossary = [
  { term: 'Alpha Power', def: 'Spectral power in 8-13 Hz band. Primary relaxation marker — decreases under anxiety.' },
  { term: 'Beta Power', def: 'Spectral power in 13-30 Hz band. Cortical arousal marker — increases during anxiety and anxiety.' },
  { term: 'Frontal Alpha Asymmetry', def: 'log(F4_alpha) - log(F3_alpha). Negative values indicate withdrawal/anxiety tendency.' },
  { term: 'Alpha/Beta Ratio', def: 'Ratio of alpha to beta power. Lower ratios indicate higher anxiety levels.' },
  { term: 'PSD', def: 'Power Spectral Density — distribution of signal power across frequencies (Welch method).' },
  { term: 'Sample Entropy', def: 'Nonlinear measure of signal complexity and regularity.' },
  { term: 'DFA', def: 'Detrended Fluctuation Analysis — characterizes long-range temporal correlations in EEG.' },
  { term: 'Hjorth Parameters', def: 'Activity (variance), Mobility (mean frequency), Complexity (bandwidth) of the signal.' },
  { term: 'SMOTE', def: 'Synthetic Minority Over-sampling Technique — generates synthetic samples for class balancing.' },
  { term: 'ICA', def: 'Independent Component Analysis — separates EEG into independent sources; used to remove artifacts.' },
  { term: "Cohen's Kappa", def: 'Agreement metric that accounts for chance agreement. Values 0.6-0.8 = substantial agreement.' },
  { term: 'MCC', def: 'Matthews Correlation Coefficient — balanced metric even with imbalanced classes. Range [-1, 1].' },
];

export default function ResearchPage() {
  const [hoveredElectrode, setHoveredElectrode] = useState<string | null>(null);

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold mb-2">
          <span className="bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">Research Background</span>
        </h1>
        <p className="text-slate-400 mb-10">Understanding the science behind EEG-based anxiety detection.</p>

        {/* Timeline */}
        <section className="mb-16">
          <h2 className="text-xl font-bold text-white mb-6">Research Timeline</h2>
          <div className="relative">
            <div className="absolute left-4 top-0 bottom-0 w-px bg-white/10" />
            <div className="space-y-6">
              {timeline.map((item, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="flex items-start gap-4 ml-0"
                >
                  <div className="w-8 h-8 rounded-full border-2 flex items-center justify-center flex-shrink-0 z-10 bg-[#0f172a]"
                    style={{ borderColor: item.color }}>
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                  </div>
                  <div className="flex-1 p-4 rounded-xl bg-white/[0.02] border border-white/5">
                    <span className="text-xs font-mono" style={{ color: item.color }}>{item.year}</span>
                    <h3 className="text-sm font-medium text-white mt-0.5">{item.title}</h3>
                    <p className="text-xs text-slate-400 mt-1">{item.desc}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* EEG Bands */}
        <section className="mb-16">
          <h2 className="text-xl font-bold text-white mb-6">EEG Bands & Anxiety</h2>
          <div className="space-y-3">
            {bands.map((band) => (
              <div key={band.name} className="p-4 rounded-xl bg-white/[0.02] border border-white/5 flex items-start gap-4">
                <div className="w-3 h-3 rounded-full mt-1 flex-shrink-0" style={{ backgroundColor: band.color }} />
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-1">
                    <span className="font-medium text-white">{band.name}</span>
                    <span className="text-xs font-mono text-slate-500">{band.range}</span>
                  </div>
                  <p className="text-xs text-slate-400"><strong className="text-slate-300">Normal:</strong> {band.normal}</p>
                  <p className="text-xs text-slate-400"><strong className="text-slate-300">Under Anxiety:</strong> {band.anxiety}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 10-20 Electrode Map */}
        <section className="mb-16">
          <h2 className="text-xl font-bold text-white mb-6">10-20 Electrode Placement</h2>
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="relative w-80 h-80">
              <svg viewBox="0 0 100 100" className="w-full h-full">
                {/* Head outline */}
                <ellipse cx="50" cy="50" rx="45" ry="48" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
                {/* Nose */}
                <path d="M 47 3 L 50 0 L 53 3" fill="none" stroke="rgba(255,255,255,0.2)" strokeWidth="0.5" />
                {/* Ears */}
                <ellipse cx="3" cy="50" rx="3" ry="6" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />
                <ellipse cx="97" cy="50" rx="3" ry="6" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="0.5" />

                {electrodes.map((e) => (
                  <g key={e.id}
                    onMouseEnter={() => setHoveredElectrode(e.id)}
                    onMouseLeave={() => setHoveredElectrode(null)}
                    className="cursor-pointer"
                  >
                    <circle
                      cx={e.cx} cy={e.cy} r={hoveredElectrode === e.id ? 3 : 2}
                      fill={hoveredElectrode === e.id ? '#6366f1' : '#22d3ee'}
                      opacity={hoveredElectrode === e.id ? 1 : 0.7}
                      className="transition-all"
                    />
                    <text x={e.cx} y={e.cy - 4} textAnchor="middle" fontSize="3"
                      fill={hoveredElectrode === e.id ? '#fff' : '#94a3b8'}
                      className="transition-all select-none"
                    >
                      {e.label}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
            <div className="flex-1 p-4 rounded-xl bg-white/[0.02] border border-white/5 min-h-[100px]">
              {hoveredElectrode ? (
                <div>
                  <span className="text-xs font-mono text-indigo-400">{hoveredElectrode}</span>
                  <p className="text-sm text-slate-300 mt-1">
                    {electrodes.find(e => e.id === hoveredElectrode)?.desc}
                  </p>
                </div>
              ) : (
                <p className="text-sm text-slate-500">Hover over an electrode to see its function</p>
              )}
            </div>
          </div>
        </section>

        {/* Dataset Comparison */}
        <section className="mb-16">
          <h2 className="text-xl font-bold text-white mb-6">Dataset Comparison</h2>
          <div className="overflow-x-auto rounded-xl bg-white/[0.02] border border-white/5">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/10">
                  {['Dataset', 'Subjects', 'EEG Ch.', 'Periph.', 'SR', 'Labels', 'Modalities', 'Reference'].map(h => (
                    <th key={h} className="text-left py-3 px-4 text-xs font-medium text-slate-400 uppercase">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasets.map(ds => (
                  <tr key={ds.name} className="border-b border-white/5 hover:bg-white/[0.02]">
                    <td className="py-3 px-4"><a href={ds.url} target="_blank" rel="noopener" className="text-indigo-400 hover:text-indigo-300 font-medium">{ds.name}</a></td>
                    <td className="py-3 px-4 text-white">{ds.subjects}</td>
                    <td className="py-3 px-4 text-white">{ds.channels}</td>
                    <td className="py-3 px-4 text-slate-400">{ds.peripheral}</td>
                    <td className="py-3 px-4 font-mono text-slate-400">{ds.sr} Hz</td>
                    <td className="py-3 px-4 text-slate-400 text-xs">{ds.labels}</td>
                    <td className="py-3 px-4 text-slate-400 text-xs max-w-[200px]">{ds.modalities}</td>
                    <td className="py-3 px-4 text-slate-500 text-xs">{ds.ref}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Methodology Flowchart */}
        <section className="mb-16">
          <h2 className="text-xl font-bold text-white mb-6">Pipeline Architecture</h2>
          <div className="flex flex-wrap items-center justify-center gap-2">
            {[
              { icon: '📡', label: 'Raw EEG', sub: 'Multi-channel signals' },
              { icon: '→', label: '', sub: '' },
              { icon: '🔧', label: 'Preprocessing', sub: 'Filter, ICA, CAR, Epoch' },
              { icon: '→', label: '', sub: '' },
              { icon: '📊', label: 'Feature Extraction', sub: '800+ features' },
              { icon: '→', label: '', sub: '' },
              { icon: '🤖', label: 'Classification', sub: '5 models' },
              { icon: '→', label: '', sub: '' },
              { icon: '✅', label: 'Anxiety Level', sub: 'Low / Moderate / High' },
            ].map((step, i) => (
              step.label ? (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="px-4 py-3 rounded-xl bg-white/[0.03] border border-white/5 text-center min-w-[120px]"
                >
                  <span className="text-2xl">{step.icon}</span>
                  <p className="text-xs font-medium text-white mt-1">{step.label}</p>
                  <p className="text-[10px] text-slate-500">{step.sub}</p>
                </motion.div>
              ) : (
                <span key={i} className="text-slate-600 text-lg">{step.icon}</span>
              )
            ))}
          </div>
        </section>

        {/* Glossary */}
        <section>
          <h2 className="text-xl font-bold text-white mb-6">Glossary of EEG Terms</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {glossary.map((item) => (
              <div key={item.term} className="p-3 rounded-xl bg-white/[0.02] border border-white/5">
                <span className="text-sm font-medium text-indigo-400">{item.term}</span>
                <p className="text-xs text-slate-400 mt-1">{item.def}</p>
              </div>
            ))}
          </div>
        </section>
      </motion.div>
    </div>
  );
}
