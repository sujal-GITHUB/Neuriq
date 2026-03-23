'use client';

import { useEffect, useRef } from 'react';
import { useTrainingStore } from '@/store/trainingStore';

export function useTrainingStatus(pollInterval: number = 2000) {
  const { jobId, isTraining, setStatus } = useTrainingStore();
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!jobId || !isTraining) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    const poll = async () => {
      try {
        const response = await fetch(`/api/training-status/${jobId}`);
        if (response.ok) {
          const status = await response.json();
          setStatus(status);
          
          if (status.status === 'completed' || status.status === 'failed') {
            if (intervalRef.current) {
              clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
          }
        }
      } catch (error) {
        console.error('Failed to poll training status:', error);
      }
    };

    // Poll immediately
    poll();
    
    // Then poll at interval
    intervalRef.current = setInterval(poll, pollInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId, isTraining, pollInterval, setStatus]);
}
