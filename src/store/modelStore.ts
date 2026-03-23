import { create } from 'zustand';
import type { ModelMetrics, ModelInfo } from '@/types';

interface ModelState {
  models: ModelInfo[];
  metrics: Record<string, ModelMetrics>;
  selectedModel: string;
  selectedDataset: string;
  setModels: (models: ModelInfo[]) => void;
  setMetrics: (key: string, metrics: ModelMetrics) => void;
  setSelectedModel: (model: string) => void;
  setSelectedDataset: (dataset: string) => void;
}

export const useModelStore = create<ModelState>((set) => ({
  models: [],
  metrics: {},
  selectedModel: 'brain2vec',
  selectedDataset: 'DEAP',
  setModels: (models) => set({ models }),
  setMetrics: (key, metrics) => set((state) => ({
    metrics: { ...state.metrics, [key]: metrics }
  })),
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSelectedDataset: (selectedDataset) => set({ selectedDataset })
}));
