import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function getAnxietyColor(level: string): string {
  switch (level) {
    case 'Low': return '#22c55e';
    case 'Moderate': return '#f59e0b';
    case 'High': return '#ef4444';
    default: return '#6366f1';
  }
}

export function getAnxietyBgClass(level: string): string {
  switch (level) {
    case 'Low': return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'Moderate': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'High': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-indigo-500/20 text-indigo-400 border-indigo-500/30';
  }
}

export const MOCK_PREDICTION: import('@/types').PredictionResult = {
  anxiety_level: 'Moderate',
  confidence: 0.847,
  probabilities: { Low: 0.12, Moderate: 0.847, High: 0.033 },
  top_features: [
    { name: 'Beta_Power_F4', value: 18.5, importance: 0.142, channel: 'F4', band: 'Beta', direction: 'positive' },
    { name: 'Alpha_Power_F3', value: 12.3, importance: 0.128, channel: 'F3', band: 'Alpha', direction: 'negative' },
    { name: 'Frontal_Asymmetry', value: -0.23, importance: 0.115, channel: 'F3-F4', band: 'Alpha', direction: 'positive' },
    { name: 'Alpha_Beta_Ratio_Cz', value: 0.67, importance: 0.098, channel: 'Cz', band: 'Alpha/Beta', direction: 'negative' },
    { name: 'Theta_Power_Fz', value: 9.8, importance: 0.091, channel: 'Fz', band: 'Theta', direction: 'positive' },
    { name: 'Sample_Entropy_C3', value: 1.34, importance: 0.084, channel: 'C3', band: 'N/A', direction: 'positive' },
    { name: 'Hjorth_Complexity_P3', value: 1.87, importance: 0.078, channel: 'P3', band: 'N/A', direction: 'negative' },
    { name: 'Gamma_Power_T7', value: 5.2, importance: 0.072, channel: 'T7', band: 'Gamma', direction: 'positive' },
    { name: 'Delta_Power_O1', value: 28.4, importance: 0.065, channel: 'O1', band: 'Delta', direction: 'negative' },
    { name: 'Spectral_Entropy_Fp1', value: 0.82, importance: 0.058, channel: 'Fp1', band: 'N/A', direction: 'positive' },
    { name: 'RMS_C4', value: 14.7, importance: 0.052, channel: 'C4', band: 'N/A', direction: 'positive' },
    { name: 'Permutation_Entropy_P4', value: 0.91, importance: 0.047, channel: 'P4', band: 'N/A', direction: 'negative' },
    { name: 'Kurtosis_F7', value: 2.3, importance: 0.043, channel: 'F7', band: 'N/A', direction: 'positive' },
    { name: 'Theta_Alpha_Ratio_Fz', value: 0.79, importance: 0.039, channel: 'Fz', band: 'Theta/Alpha', direction: 'positive' },
    { name: 'Zero_Crossing_Rate_T8', value: 52.3, importance: 0.035, channel: 'T8', band: 'N/A', direction: 'negative' },
    { name: 'DWT_Energy_D3_Cz', value: 342.1, importance: 0.031, channel: 'Cz', band: 'DWT-D3', direction: 'positive' },
    { name: 'Higuchi_FD_AF3', value: 1.52, importance: 0.028, channel: 'AF3', band: 'N/A', direction: 'positive' },
    { name: 'Peak_to_Peak_CP5', value: 67.8, importance: 0.025, channel: 'CP5', band: 'N/A', direction: 'negative' },
    { name: 'Variance_FC1', value: 45.2, importance: 0.022, channel: 'FC1', band: 'N/A', direction: 'positive' },
    { name: 'DFA_Exponent_O2', value: 0.73, importance: 0.019, channel: 'O2', band: 'N/A', direction: 'positive' },
  ],
  band_powers: { delta: 25.3, theta: 12.8, alpha: 18.5, beta: 15.2, gamma: 6.7 },
  alpha_beta_ratio: 1.217,
  frontal_asymmetry: -0.234,
  model_used: 'brain2vec',
  inference_time_ms: 42.5
};
