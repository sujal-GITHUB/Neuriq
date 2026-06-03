import { create } from 'zustand';
import type { UploadedEEG } from '@/types';

interface SignalState {
  uploadedEEG: UploadedEEG | null;
  selectedChannels: string[];
  samplingRate: number;
  setUploadedEEG: (data: UploadedEEG) => void;
  setSelectedChannels: (channels: string[]) => void;
  setSamplingRate: (rate: number) => void;
  clear: () => void;
}

export const useSignalStore = create<SignalState>((set) => ({
  uploadedEEG: null,
  selectedChannels: [],
  samplingRate: 128,
  setUploadedEEG: (data) => set({
    uploadedEEG: data,
    selectedChannels: data.channels,
    samplingRate: data.sampling_rate
  }),
  setSelectedChannels: (channels) => set({ selectedChannels: channels }),
  setSamplingRate: (rate) => set({ samplingRate: rate }),
  clear: () => set({ uploadedEEG: null, selectedChannels: [], samplingRate: 128 })
}));
