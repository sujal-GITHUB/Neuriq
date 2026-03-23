"""
NeuroAnxiety — Frequency-Domain Feature Extraction
===================================================
Extracts spectral features from EEG epochs using Welch's PSD method.

Features:
  - Absolute and relative band powers (delta, theta, alpha, beta, gamma)
  - Alpha/Beta ratio (anxiety marker)
  - Theta/Alpha ratio
  - Frontal Alpha Asymmetry
  - Spectral entropy
  - Mean and median frequency
  - Wavelet-based features (DWT)

References:
  - Gandhi & Jaliya (2024): Alpha/Beta ratio as anxiety biomarker
  - Chaudhari & Shrivastava (2024): EEG band power analysis
  - Badr et al. (2024): Spectral feature review
"""

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Optional
import pywt

from config import FREQ_BANDS, SAMPLING_RATE


def extract_frequency_features(
    signal: np.ndarray,
    sr: int = SAMPLING_RATE,
    channel_name: str = "",
    nperseg: int = 256
) -> Dict[str, float]:
    """
    Extract frequency-domain features from a single-channel epoch.
    
    Args:
        signal: 1D array (n_samples,)
        sr: sampling rate
        channel_name: channel identifier
        nperseg: Welch window size
        
    Returns:
        Dict of feature_name -> value
    """
    prefix = f"{channel_name}_" if channel_name else ""
    features = {}
    
    # Compute PSD using Welch's method
    nperseg_actual = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=sr, nperseg=nperseg_actual)
    
    # Total power
    total_power = np.sum(psd)
    if total_power < 1e-20:
        total_power = 1e-20
    
    # Band powers (absolute and relative)
    band_powers = {}
    for band_name, (low, high) in FREQ_BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        abs_power = float(np.sum(psd[mask])) if np.any(mask) else 0.0
        rel_power = abs_power / total_power
        
        band_powers[band_name] = abs_power
        features[f"{prefix}{band_name}_abs_power"] = abs_power
        features[f"{prefix}{band_name}_rel_power"] = rel_power
    
    # Alpha/Beta ratio (anxiety marker: decreases under anxiety)
    alpha_power = band_powers.get('alpha', 1e-10)
    beta_power = band_powers.get('beta', 1e-10)
    features[f"{prefix}alpha_beta_ratio"] = float(alpha_power / max(beta_power, 1e-10))
    
    # Theta/Alpha ratio
    theta_power = band_powers.get('theta', 1e-10)
    features[f"{prefix}theta_alpha_ratio"] = float(theta_power / max(alpha_power, 1e-10))
    
    # Spectral entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    features[f"{prefix}spectral_entropy"] = float(scipy_entropy(psd_norm, base=2))
    
    # Mean frequency
    features[f"{prefix}mean_frequency"] = float(np.sum(freqs * psd) / total_power)
    
    # Median frequency
    cumulative_power = np.cumsum(psd)
    median_idx = np.searchsorted(cumulative_power, total_power / 2)
    features[f"{prefix}median_frequency"] = float(freqs[min(median_idx, len(freqs) - 1)])
    
    return features


def compute_frontal_alpha_asymmetry(
    epoch: np.ndarray,
    channel_names: List[str],
    sr: int = SAMPLING_RATE
) -> Dict[str, float]:
    """
    Compute Frontal Alpha Asymmetry: log(F4_alpha) - log(F3_alpha).
    Positive values → relatively more left frontal activity → approach motivation.
    Negative values → withdrawal/anxiety.
    
    Args:
        epoch: shape (n_channels, n_samples)
        channel_names: list of channel names
        sr: sampling rate
    """
    features = {}
    
    frontal_pairs = [('F3', 'F4'), ('Fp1', 'Fp2'), ('AF3', 'AF4')]
    
    for left_ch, right_ch in frontal_pairs:
        if left_ch in channel_names and right_ch in channel_names:
            left_idx = channel_names.index(left_ch)
            right_idx = channel_names.index(right_ch)
            
            # Compute alpha power for each
            nperseg = min(256, epoch.shape[1])
            freqs_l, psd_l = welch(epoch[left_idx], fs=sr, nperseg=nperseg)
            freqs_r, psd_r = welch(epoch[right_idx], fs=sr, nperseg=nperseg)
            
            alpha_band = FREQ_BANDS['alpha']
            mask_l = (freqs_l >= alpha_band[0]) & (freqs_l <= alpha_band[1])
            mask_r = (freqs_r >= alpha_band[0]) & (freqs_r <= alpha_band[1])
            
            alpha_l = max(float(np.sum(psd_l[mask_l])), 1e-20)
            alpha_r = max(float(np.sum(psd_r[mask_r])), 1e-20)
            
            asymmetry = np.log(alpha_r) - np.log(alpha_l)
            features[f"frontal_asymmetry_{left_ch}_{right_ch}"] = float(asymmetry)
    
    # Primary asymmetry (F3/F4 if available)
    if 'frontal_asymmetry_F3_F4' in features:
        features['frontal_alpha_asymmetry'] = features['frontal_asymmetry_F3_F4']
    elif features:
        features['frontal_alpha_asymmetry'] = float(list(features.values())[0])
    else:
        features['frontal_alpha_asymmetry'] = 0.0
    
    return features


def extract_wavelet_features(
    signal: np.ndarray,
    channel_name: str = "",
    wavelet: str = 'db4',
    level: int = 4
) -> Dict[str, float]:
    """
    Extract wavelet features using Discrete Wavelet Transform (DWT).
    
    Args:
        signal: 1D array (n_samples,)
        channel_name: channel identifier
        wavelet: wavelet type
        level: decomposition level
        
    Returns:
        Energy and entropy of each sub-band
    """
    prefix = f"{channel_name}_" if channel_name else ""
    features = {}
    
    # DWT decomposition
    max_level = pywt.dwt_max_level(len(signal), wavelet)
    actual_level = min(level, max_level)
    
    if actual_level < 1:
        return features
    
    coeffs = pywt.wavedec(signal, wavelet, level=actual_level)
    
    for i, coeff in enumerate(coeffs):
        band_label = f"A{actual_level}" if i == 0 else f"D{actual_level - i + 1}"
        
        # Energy
        energy = float(np.sum(coeff ** 2))
        features[f"{prefix}dwt_energy_{band_label}"] = energy
        
        # Entropy
        coeff_abs = np.abs(coeff)
        coeff_norm = coeff_abs / max(np.sum(coeff_abs), 1e-20)
        coeff_norm = coeff_norm[coeff_norm > 0]
        ent = float(scipy_entropy(coeff_norm, base=2))
        features[f"{prefix}dwt_entropy_{band_label}"] = ent
    
    return features


def extract_frequency_features_multichannel(
    epoch: np.ndarray,
    channel_names: List[str],
    sr: int = SAMPLING_RATE
) -> Dict[str, float]:
    """
    Extract frequency features from all channels + cross-channel features.
    
    Args:
        epoch: shape (n_channels, n_samples)
        channel_names: list of channel names
        sr: sampling rate
        
    Returns:
        Combined feature dict
    """
    all_features = {}
    
    for ch_idx, ch_name in enumerate(channel_names):
        # Spectral features
        features = extract_frequency_features(epoch[ch_idx], sr, ch_name)
        all_features.update(features)
        
        # Wavelet features
        wavelet_features = extract_wavelet_features(epoch[ch_idx], ch_name)
        all_features.update(wavelet_features)
    
    # Frontal alpha asymmetry
    asymmetry_features = compute_frontal_alpha_asymmetry(epoch, channel_names, sr)
    all_features.update(asymmetry_features)
    
    return all_features
