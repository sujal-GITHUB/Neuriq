"""
NeuroAnxiety — EEG Preprocessing Pipeline
==========================================
Implements a complete preprocessing chain for raw EEG signals.

Pipeline (in order):
  1. Downsampling → 128 Hz
  2. Bandpass filtering → 0.5–45 Hz (Butterworth, order=5)
  3. Notch filter → 50 Hz
  4. ICA artifact removal (FastICA, n=20)
  5. Common Average Reference (CAR)
  6. Baseline correction (first 3 seconds)
  7. Epoching → 4-second non-overlapping windows
  8. Z-score normalization per channel per trial
  9. SMOTE oversampling on training set

References:
  - Chaudhari & Shrivastava (2024): preprocessing best practices for EEG anxiety detection
  - Badr et al. (2024): review of preprocessing pipelines for DL-based EEG analysis
"""

import numpy as np
import logging
from typing import Tuple, List, Optional
from scipy.signal import butter, filtfilt, iirnotch, resample

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Complete EEG signal preprocessing pipeline."""
    
    def __init__(
        self,
        target_sr: int = 128,
        bandpass_low: float = 0.5,
        bandpass_high: float = 45.0,
        notch_freq: float = 50.0,
        filter_order: int = 5,
        ica_n_components: int = 20,
        epoch_duration: float = 4.0,
        baseline_duration: float = 3.0,
        use_ica: bool = True
    ):
        self.target_sr = target_sr
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        self.ica_n_components = ica_n_components
        self.epoch_duration = epoch_duration
        self.epoch_samples = int(target_sr * epoch_duration)
        self.baseline_duration = baseline_duration
        self.baseline_samples = int(target_sr * baseline_duration)
        self.use_ica = use_ica
    
    def process(
        self,
        data: np.ndarray,
        original_sr: int,
        channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Run the full preprocessing pipeline on raw EEG data.
        
        Args:
            data: shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)
            original_sr: original sampling rate
            channel_names: optional channel names for ICA
            
        Returns:
            Preprocessed data, same shape convention
        """
        is_3d = data.ndim == 3
        
        if is_3d:
            processed_trials = []
            for trial_idx in range(data.shape[0]):
                trial = data[trial_idx]
                processed = self._process_single(trial, original_sr, channel_names)
                processed_trials.append(processed)
            return np.array(processed_trials)
        else:
            return self._process_single(data, original_sr, channel_names)
    
    def _process_single(
        self,
        data: np.ndarray,
        original_sr: int,
        channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Process a single trial/recording: (n_channels, n_samples)."""
        
        # Step 1: Downsample
        if original_sr != self.target_sr:
            data = self._downsample(data, original_sr)
            logger.debug(f"Downsampled: {original_sr} Hz → {self.target_sr} Hz")
        
        # Step 2: Bandpass filter
        data = self._bandpass_filter(data)
        
        # Step 3: Notch filter
        data = self._notch_filter(data)
        
        # Step 4: ICA artifact removal
        if self.use_ica and data.shape[0] >= self.ica_n_components:
            try:
                data = self._ica_artifact_removal(data, channel_names)
            except Exception as e:
                logger.warning(f"ICA failed, skipping: {e}")
        
        # Step 5: Common Average Reference
        data = self._common_average_reference(data)
        
        # Step 6: Baseline correction
        data = self._baseline_correction(data)
        
        # Step 8: Z-score normalization per channel
        data = self._normalize(data)
        
        return data
    
    def epoch(self, data: np.ndarray) -> np.ndarray:
        """
        Step 7: Segment continuous data into fixed-length epochs.
        
        Args:
            data: (n_channels, n_samples) preprocessed data
            
        Returns:
            (n_epochs, n_channels, epoch_samples)
        """
        n_channels, n_samples = data.shape
        n_epochs = n_samples // self.epoch_samples
        
        if n_epochs == 0:
            logger.warning(
                f"Signal too short for epoching: {n_samples} samples < "
                f"{self.epoch_samples} epoch samples. Returning padded single epoch."
            )
            padded = np.zeros((n_channels, self.epoch_samples))
            padded[:, :n_samples] = data
            return padded[np.newaxis, :, :]
        
        epochs = np.zeros((n_epochs, n_channels, self.epoch_samples))
        for i in range(n_epochs):
            start = i * self.epoch_samples
            end = start + self.epoch_samples
            epochs[i] = data[:, start:end]
        
        return epochs
    
    def _downsample(self, data: np.ndarray, original_sr: int) -> np.ndarray:
        """Step 1: Resample to target sampling rate."""
        n_target_samples = int(data.shape[1] * self.target_sr / original_sr)
        return resample(data, n_target_samples, axis=1)
    
    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Step 2: Butterworth bandpass filter."""
        nyquist = self.target_sr / 2.0
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        
        # Clamp values to valid range
        low = max(low, 0.001)
        high = min(high, 0.999)
        
        b, a = butter(self.filter_order, [low, high], btype='band')
        
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            try:
                filtered[ch] = filtfilt(b, a, data[ch])
            except ValueError:
                filtered[ch] = data[ch]
        
        return filtered
    
    def _notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Step 3: Notch filter for power-line noise removal."""
        quality_factor = 30.0
        b, a = iirnotch(self.notch_freq, quality_factor, self.target_sr)
        
        filtered = np.zeros_like(data)
        for ch in range(data.shape[0]):
            try:
                filtered[ch] = filtfilt(b, a, data[ch])
            except ValueError:
                filtered[ch] = data[ch]
        
        return filtered
    
    def _ica_artifact_removal(
        self,
        data: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Step 4: ICA-based automatic artifact removal using MNE."""
        import mne
        
        n_components = min(self.ica_n_components, data.shape[0])
        
        # Create MNE Raw object
        ch_names = channel_names or [f'EEG{i+1}' for i in range(data.shape[0])]
        ch_types = ['eeg'] * data.shape[0]
        info = mne.create_info(ch_names=ch_names[:data.shape[0]], sfreq=self.target_sr, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        # Fit ICA
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method='fastica',
            random_state=42,
            verbose=False
        )
        ica.fit(raw, verbose=False)
        
        # Auto-detect EOG and muscle artifacts
        try:
            eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
            ica.exclude = eog_indices[:3]  # Remove up to 3 components
        except Exception:
            # If no EOG channels, exclude components with high variance
            variances = np.var(ica.get_sources(raw).get_data(), axis=1)
            threshold = np.mean(variances) + 2 * np.std(variances)
            ica.exclude = [i for i, v in enumerate(variances) if v > threshold][:3]
        
        # Apply ICA
        raw_clean = ica.apply(raw, verbose=False)
        
        return raw_clean.get_data()
    
    def _common_average_reference(self, data: np.ndarray) -> np.ndarray:
        """Step 5: Common Average Reference (CAR)."""
        mean_ref = np.mean(data, axis=0, keepdims=True)
        return data - mean_ref
    
    def _baseline_correction(self, data: np.ndarray) -> np.ndarray:
        """Step 6: Subtract mean of pre-stimulus baseline (first 3 seconds)."""
        baseline_end = min(self.baseline_samples, data.shape[1])
        if baseline_end > 0:
            baseline_mean = np.mean(data[:, :baseline_end], axis=1, keepdims=True)
            data = data - baseline_mean
        return data
    
    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Step 8: Z-score normalization per channel."""
        for ch in range(data.shape[0]):
            mean = np.mean(data[ch])
            std = np.std(data[ch])
            if std > 1e-10:
                data[ch] = (data[ch] - mean) / std
            else:
                data[ch] = data[ch] - mean
        return data
    
    @staticmethod
    def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 9: SMOTE oversampling for class imbalance.
        Applied only on training set.
        
        Args:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,)
            
        Returns:
            Resampled (X, y)
        """
        from imblearn.over_sampling import SMOTE
        
        # Need at least k_neighbors+1 samples per class
        min_class_count = min(np.bincount(y))
        k = min(5, min_class_count - 1) if min_class_count > 1 else 1
        
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(
            f"SMOTE: {len(y)} → {len(y_resampled)} samples. "
            f"Distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}"
        )
        
        return X_resampled, y_resampled
