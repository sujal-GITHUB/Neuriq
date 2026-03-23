import axios from 'axios';
import type {
  PredictionResult, TrainingRequest, TrainingStatus,
  ModelInfo, ModelMetrics, DatasetInfo, UploadedEEG
} from '@/types';

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' }
});

export const predictFromSignals = async (
  signals: number[][],
  channels: string[],
  samplingRate: number,
  model: string = 'brain2vec'
): Promise<PredictionResult> => {
  const { data } = await api.post('/predict', {
    signals, channels, sampling_rate: samplingRate,
    modalities: ['EEG'], model
  });
  return data;
};

export const predictFromFeatures = async (
  features: Record<string, number>,
  model: string = 'ensemble'
): Promise<PredictionResult> => {
  const { data } = await api.post('/predict', { features, model, manual: true });
  return data;
};

export const startTraining = async (
  request: TrainingRequest
): Promise<{ status: string; job_id: string }> => {
  const { data } = await api.post('/train', request);
  return data;
};

export const getTrainingStatus = async (
  jobId: string
): Promise<TrainingStatus> => {
  const { data } = await api.get(`/training-status/${jobId}`);
  return data;
};

export const getMetrics = async (
  model: string, dataset: string
): Promise<ModelMetrics> => {
  const { data } = await api.get(`/metrics/${model}/${dataset}`);
  return data;
};

export const listModels = async (): Promise<{ models: ModelInfo[] }> => {
  const { data } = await api.get('/predict'); // Uses mock for now
  // In production, this calls /models
  return data;
};

export const uploadEEGFile = async (
  file: File
): Promise<UploadedEEG> => {
  const formData = new FormData();
  formData.append('file', file);
  const { data } = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return data;
};

export default api;
