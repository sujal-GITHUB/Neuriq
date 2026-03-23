"""
NeuroAnxiety — Time-Domain Feature Extraction
==============================================
Extracts time-domain features from EEG epochs.

Features:
  - Mean, variance, standard deviation
  - Skewness, kurtosis
  - Zero-crossing rate
  - Peak-to-peak amplitude
  - Hjorth parameters: Activity, Mobility, Complexity
  - Root Mean Square (RMS)

Reference: Gandhi & Jaliya (2024)
"""

import numpy as np
from scipy import stats
from typing import Dict, List


def extract_time_features(signal: np.ndarray, channel_name: str = "") -> Dict[str, float]:
    """
    Extract all time-domain features from a single-channel epoch.
    
    Args:
        signal: 1D array of shape (n_samples,)
        channel_name: channel identifier for feature naming
        
    Returns:
        Dict of feature_name -> value
    """
    prefix = f"{channel_name}_" if channel_name else ""
    features = {}
    
    # Basic statistics
    features[f"{prefix}mean"] = float(np.mean(signal))
    features[f"{prefix}variance"] = float(np.var(signal))
    features[f"{prefix}std"] = float(np.std(signal))
    
    # Higher-order statistics
    features[f"{prefix}skewness"] = float(stats.skew(signal))
    features[f"{prefix}kurtosis"] = float(stats.kurtosis(signal))
    
    # Zero-crossing rate
    features[f"{prefix}zero_crossing_rate"] = float(_zero_crossing_rate(signal))
    
    # Peak-to-peak amplitude
    features[f"{prefix}peak_to_peak"] = float(np.ptp(signal))
    
    # Hjorth parameters
    activity, mobility, complexity = _hjorth_parameters(signal)
    features[f"{prefix}hjorth_activity"] = activity
    features[f"{prefix}hjorth_mobility"] = mobility
    features[f"{prefix}hjorth_complexity"] = complexity
    
    # Root Mean Square
    features[f"{prefix}rms"] = float(np.sqrt(np.mean(signal ** 2)))
    
    return features


def _zero_crossing_rate(signal: np.ndarray) -> float:
    """Count zero crossings normalized by signal length."""
    zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
    return zero_crossings / len(signal)


def _hjorth_parameters(signal: np.ndarray) -> tuple:
    """
    Compute Hjorth parameters: Activity, Mobility, Complexity.
    
    Activity = variance of the signal
    Mobility = sqrt(var(derivative) / var(signal))
    Complexity = Mobility(derivative) / Mobility(signal)
    """
    # Activity
    activity = float(np.var(signal))
    
    # First derivative
    d1 = np.diff(signal)
    
    # Second derivative
    d2 = np.diff(d1)
    
    # Mobility
    var_d1 = np.var(d1)
    if activity > 1e-10:
        mobility = float(np.sqrt(var_d1 / activity))
    else:
        mobility = 0.0
    
    # Complexity
    var_d2 = np.var(d2)
    if var_d1 > 1e-10:
        mobility_d1 = np.sqrt(var_d2 / var_d1)
        if mobility > 1e-10:
            complexity = float(mobility_d1 / mobility)
        else:
            complexity = 0.0
    else:
        complexity = 0.0
    
    return activity, mobility, complexity


def extract_time_features_multichannel(
    epoch: np.ndarray,
    channel_names: List[str]
) -> Dict[str, float]:
    """
    Extract time-domain features from all channels of an epoch.
    
    Args:
        epoch: shape (n_channels, n_samples)
        channel_names: list of channel names
        
    Returns:
        Combined feature dict
    """
    all_features = {}
    
    for ch_idx, ch_name in enumerate(channel_names):
        features = extract_time_features(epoch[ch_idx], ch_name)
        all_features.update(features)
    
    return all_features
