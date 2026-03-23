/**
 * Generate a realistic 30-second, 14-channel EEG CSV at 128 Hz.
 * Channels: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4
 * Synthetic data: alpha (10 Hz) + beta (20 Hz) + noise, 5-100 µV range.
 * 
 * Run: node scripts/generate-sample-eeg.js
 */

const fs = require('fs');
const path = require('path');

const SR = 128;
const DURATION = 30;
const N = SR * DURATION;
const channels = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4'];

let csv = 'timestamp,' + channels.join(',') + '\n';

for (let i = 0; i < N; i++) {
  const t = i / SR;
  const row = [t.toFixed(4)];
  
  for (let ch = 0; ch < channels.length; ch++) {
    // Channel-specific phase offsets
    const phaseAlpha = ch * 0.5;
    const phaseBeta = ch * 0.3;
    
    // Alpha oscillations (10 Hz) — dominant in relaxed frontal/parietal
    const alphaAmp = (ch < 4 || ch >= 8) ? 20 : 12; // stronger in frontal/occipital
    const alpha = alphaAmp * Math.sin(2 * Math.PI * 10 * t + phaseAlpha);
    
    // Beta oscillations (20 Hz) — present during active cognition
    const betaAmp = (ch < 4) ? 10 : 6;
    const beta = betaAmp * Math.sin(2 * Math.PI * 22 * t + phaseBeta);
    
    // Theta (5 Hz) — low amplitude background
    const theta = 5 * Math.sin(2 * Math.PI * 5 * t + ch * 0.7);
    
    // Delta (2 Hz) — slow wave
    const delta = 8 * Math.sin(2 * Math.PI * 2 * t + ch * 0.2);
    
    // Pink noise approximation
    const noise = (Math.random() - 0.5) * 8;
    
    // EMG artifact (occasional)
    const emg = (Math.random() > 0.98) ? (Math.random() - 0.5) * 30 : 0;
    
    const value = alpha + beta + theta + delta + noise + emg;
    row.push(value.toFixed(2));
  }
  
  csv += row.join(',') + '\n';
}

const outPath = path.join(__dirname, '..', 'public', 'sample-eeg.csv');
fs.writeFileSync(outPath, csv);
console.log(`Generated ${outPath}: ${N} samples, ${channels.length} channels, ${DURATION}s at ${SR}Hz`);
