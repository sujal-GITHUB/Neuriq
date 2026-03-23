"""
NeuroAnxiety ML Service — Inference Module
==========================================
Handles inference logic: preprocessing input, running models, formatting output.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional

from config import FREQ_BANDS, ANXIETY_LABELS, SAMPLING_RATE


class AnxietyInferenceEngine:
    """Unified inference engine for all model types."""
    
    def __init__(self):
        self.loaded_models = {}
        self.preprocessor = None
        self.feature_extractor = None
    
    def load_model(self, model_name: str, checkpoint_path: str = None):
        """Load a trained model from checkpoint."""
        # TODO: Implement real model loading in Step 12
        pass
    
    def predict_from_signals(
        self,
        signals: np.ndarray,
        channels: List[str],
        sampling_rate: int,
        model_name: str = "brain2vec"
    ) -> Dict[str, Any]:
        """
        Run full inference pipeline on raw EEG signals.
        
        Args:
            signals: shape (n_channels, n_samples)
            channels: list of channel names
            sampling_rate: original sampling rate
            model_name: which model to use
            
        Returns:
            Prediction results with anxiety level, confidence, features, etc.
        """
        start_time = time.time()
        
        # TODO: Implement real pipeline in Step 12
        # 1. Preprocess signals
        # 2. Extract features
        # 3. Run model inference
        # 4. Compute explainability
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "anxiety_level": "Moderate",
            "confidence": 0.85,
            "probabilities": {"Low": 0.10, "Moderate": 0.85, "High": 0.05},
            "inference_time_ms": round(inference_time, 2)
        }
    
    def predict_from_features(
        self,
        features: Dict[str, float],
        model_name: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Run inference from pre-computed features (manual input).
        
        Args:
            features: dict of feature_name -> value
            model_name: which model to use
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        # TODO: Implement real inference in Step 12
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            "anxiety_level": "Low",
            "confidence": 0.72,
            "probabilities": {"Low": 0.72, "Moderate": 0.20, "High": 0.08},
            "inference_time_ms": round(inference_time, 2)
        }
    
    def compute_band_powers(self, signals: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute average band powers across channels."""
        from scipy.signal import welch
        
        band_powers = {}
        for band_name, (low, high) in FREQ_BANDS.items():
            powers = []
            for ch_signal in signals:
                freqs, psd = welch(ch_signal, fs=sr, nperseg=min(256, len(ch_signal)))
                band_mask = (freqs >= low) & (freqs <= high)
                powers.append(np.mean(psd[band_mask]) if np.any(band_mask) else 0.0)
            band_powers[band_name] = round(float(np.mean(powers)), 4)
        
        return band_powers
