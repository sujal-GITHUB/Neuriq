// NeuroAnxiety — TypeScript Type Definitions

export interface PredictionResult {
  anxiety_level: 'Low' | 'Moderate' | 'High';
  confidence: number;
  probabilities: {
    Low: number;
    Moderate: number;
    High: number;
  };
  top_features: FeatureImportance[];
  band_powers: BandPowers;
  alpha_beta_ratio: number;
  frontal_asymmetry: number;
  model_used: string;
  inference_time_ms: number;
}

export interface FeatureImportance {
  name: string;
  value?: number;
  importance: number;
  channel?: string;
  band?: string;
  direction?: 'positive' | 'negative';
  mean_shap?: number;
}

export interface BandPowers {
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
}

export interface TrainingRequest {
  dataset: 'DEAP' | 'DREAMER' | 'MAHNOB';
  model: 'boosting_ensemble' | 'ensemble' | 'brain2vec' | 'cnn_lstm' | 'eegnet';
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  cv_folds?: number;
  use_smote?: boolean;
  eval_split?: 'kfold' | 'loso';
}

export interface TrainingStatus {
  status: 'running' | 'completed' | 'failed';
  progress: number;
  current_epoch: number;
  total_epochs: number;
  current_metrics: {
    train_loss: number;
    val_loss: number;
    train_accuracy: number;
    val_accuracy: number;
    learning_rate: number;
  };
  log?: string[];
}

export interface ModelInfo {
  name: string;
  display_name: string;
  type: 'deep' | 'classical';
  framework: string;
  parameters: number | null;
  description: string;
  trained_on: string[];
  best_accuracy: number;
  checkpoint_path: string;
}

export interface ModelMetrics {
  model_name: string;
  dataset: string;
  per_class: {
    precision: Record<string, number>;
    recall: Record<string, number>;
    f1_score: Record<string, number>;
    support: Record<string, number>;
    specificity: Record<string, number>;
  };
  overall: {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    cohens_kappa: number;
    mcc: number;
    balanced_accuracy: number;
    roc_auc?: Record<string, number>;
    average_precision?: Record<string, number>;
  };
  confusion_matrix: {
    raw: number[][];
    normalized: number[][];
    labels: string[];
  };
  roc_curves?: Record<string, { fpr: number[]; tpr: number[] }>;
  cross_validation: {
    fold_accuracies: number[];
    mean_accuracy: number;
    std_accuracy: number;
    fold_f1s: number[];
    training_times: number[];
    inference_time_ms: number;
  };
  statistical_tests: {
    mcnemar: { statistic: number; p_value: number };
    friedman: { statistic: number; p_value: number };
  };
  shap?: {
    top_features: FeatureImportance[];
  };
}

export interface DatasetInfo {
  name: string;
  full_name: string;
  loaded: boolean;
  num_subjects: number;
  num_trials: number;
  num_eeg_channels: number;
  num_peripheral_channels: number;
  sampling_rate: number;
  trial_duration_sec: number;
  labels: string[];
  label_scale: string;
  modalities: string[];
  format: string;
  download_url: string;
  reference: string;
}

export interface UploadedEEG {
  filename: string;
  channels: string[];
  sampling_rate: number;
  duration_sec: number;
  num_samples: number;
  signals: number[][];
}

export interface ManualFeatures {
  delta_power: number;
  theta_power: number;
  alpha_power: number;
  beta_power: number;
  gamma_power: number;
  frontal_asymmetry: number;
  sample_entropy: number;
  approx_entropy: number;
  hjorth_complexity: number;
  gsr_mean?: number;
  gsr_variance?: number;
  heart_rate?: number;
  hrv_rmssd?: number;
  skin_temperature?: number;
  emg_amplitude?: number;
}
