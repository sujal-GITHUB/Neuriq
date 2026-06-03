import { create } from 'zustand';
import type { TrainingStatus } from '@/types';

interface TrainingState {
  jobId: string | null;
  status: TrainingStatus | null;
  isTraining: boolean;
  setJobId: (id: string) => void;
  setStatus: (status: TrainingStatus) => void;
  setIsTraining: (training: boolean) => void;
  clear: () => void;
}

export const useTrainingStore = create<TrainingState>((set) => ({
  jobId: null,
  status: null,
  isTraining: false,
  setJobId: (jobId) => set({ jobId, isTraining: true }),
  setStatus: (status) => set({ status, isTraining: status.status === 'running' }),
  setIsTraining: (isTraining) => set({ isTraining }),
  clear: () => set({ jobId: null, status: null, isTraining: false })
}));
