import { create } from 'zustand';
import type { PredictionResult } from '@/types';

interface ResultState {
  result: PredictionResult | null;
  isLoading: boolean;
  error: string | null;
  setResult: (result: PredictionResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clear: () => void;
}

export const useResultStore = create<ResultState>((set) => ({
  result: null,
  isLoading: false,
  error: null,
  setResult: (result) => set({ result, isLoading: false, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error, isLoading: false }),
  clear: () => set({ result: null, isLoading: false, error: null })
}));
