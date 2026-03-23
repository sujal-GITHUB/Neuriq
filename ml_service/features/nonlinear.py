"""
NeuroAnxiety — Nonlinear Feature Extraction
============================================
Entropy and fractal dimension features using the antropy library.

Features:
  - Sample Entropy (SampEn)
  - Approximate Entropy (ApEn)
  - Permutation Entropy
  - Spectral Entropy
  - Hjorth Fractal Dimension
  - Higuchi Fractal Dimension
  - Detrended Fluctuation Analysis (DFA) exponent

Reference: Badr et al. (2024) — nonlinear EEG features for anxiety detection
"""

import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_nonlinear_features(
    signal: np.ndarray,
    channel_name: str = "",
    sr: int = 128
) -> Dict[str, float]:
    """
    Extract nonlinear features from a single-channel epoch.
    
    Args:
        signal: 1D array (n_samples,)
        channel_name: channel identifier
        sr: sampling rate
        
    Returns:
        Dict of feature_name -> value
    """
    import antropy as ant
    
    prefix = f"{channel_name}_" if channel_name else ""
    features = {}
    
    # Ensure signal is float64 for numerical stability
    signal = signal.astype(np.float64)
    
    # Sample Entropy
    try:
        features[f"{prefix}sample_entropy"] = float(ant.sample_entropy(signal))
    except Exception:
        features[f"{prefix}sample_entropy"] = 0.0
    
    # Approximate Entropy
    try:
        features[f"{prefix}approx_entropy"] = float(ant.app_entropy(signal))
    except Exception:
        features[f"{prefix}approx_entropy"] = 0.0
    
    # Permutation Entropy
    try:
        features[f"{prefix}permutation_entropy"] = float(
            ant.perm_entropy(signal, order=3, delay=1, normalize=True)
        )
    except Exception:
        features[f"{prefix}permutation_entropy"] = 0.0
    
    # Spectral Entropy
    try:
        features[f"{prefix}spectral_entropy"] = float(
            ant.spectral_entropy(signal, sf=sr, method='welch', normalize=True)
        )
    except Exception:
        features[f"{prefix}spectral_entropy"] = 0.0
    
    # Hjorth Fractal Dimension
    try:
        features[f"{prefix}hjorth_fd"] = float(ant.hjorth_params(signal)[1])
    except Exception:
        features[f"{prefix}hjorth_fd"] = 0.0
    
    # Higuchi Fractal Dimension
    try:
        features[f"{prefix}higuchi_fd"] = float(ant.higuchi_fd(signal))
    except Exception:
        features[f"{prefix}higuchi_fd"] = 0.0
    
    # Detrended Fluctuation Analysis
    try:
        features[f"{prefix}dfa"] = float(ant.detrended_fluctuation(signal))
    except Exception:
        features[f"{prefix}dfa"] = 0.0
    
    return features


def extract_nonlinear_features_multichannel(
    epoch: np.ndarray,
    channel_names: List[str],
    sr: int = 128
) -> Dict[str, float]:
    """
    Extract nonlinear features from all channels of an epoch.
    
    Args:
        epoch: shape (n_channels, n_samples)
        channel_names: list of channel names
        sr: sampling rate
        
    Returns:
        Combined feature dict
    """
    all_features = {}
    
    for ch_idx, ch_name in enumerate(channel_names):
        features = extract_nonlinear_features(epoch[ch_idx], ch_name, sr)
        all_features.update(features)
    
    return all_features
