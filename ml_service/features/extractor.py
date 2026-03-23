"""
NeuroAnxiety — Unified Feature Extraction Orchestrator
======================================================
Combines time-domain, frequency-domain, and nonlinear features.
Applies feature selection (variance threshold + SelectKBest).

Expected output: ~500-800 features per epoch → reduced to 200 via selection.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

from features.time_domain import extract_time_features_multichannel
from features.frequency_domain import extract_frequency_features_multichannel
from features.nonlinear import extract_nonlinear_features_multichannel
from config import SAMPLING_RATE, VARIANCE_THRESHOLD, FEATURE_SELECTION_K

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Orchestrates extraction of all feature types from EEG epochs."""
    
    def __init__(
        self,
        sr: int = SAMPLING_RATE,
        variance_threshold: float = VARIANCE_THRESHOLD,
        select_k: int = FEATURE_SELECTION_K,
        use_nonlinear: bool = True
    ):
        self.sr = sr
        self.variance_threshold = variance_threshold
        self.select_k = select_k
        self.use_nonlinear = use_nonlinear
        
        self.var_selector = None
        self.kbest_selector = None
        self.feature_names = None
        self.selected_feature_names = None
        self._is_fitted = False
    
    def extract_single_epoch(
        self,
        epoch: np.ndarray,
        channel_names: List[str]
    ) -> Dict[str, float]:
        """
        Extract all features from a single epoch.
        
        Args:
            epoch: shape (n_channels, n_samples)
            channel_names: list of channel names
            
        Returns:
            Dict of feature_name -> value
        """
        all_features = {}
        
        # Time-domain features
        time_features = extract_time_features_multichannel(epoch, channel_names)
        all_features.update(time_features)
        
        # Frequency-domain features (including wavelet and frontal asymmetry)
        freq_features = extract_frequency_features_multichannel(epoch, channel_names, self.sr)
        all_features.update(freq_features)
        
        # Nonlinear features
        if self.use_nonlinear:
            try:
                nonlinear_features = extract_nonlinear_features_multichannel(
                    epoch, channel_names, self.sr
                )
                all_features.update(nonlinear_features)
            except Exception as e:
                logger.warning(f"Nonlinear feature extraction failed: {e}")
        
        return all_features
    
    def extract_dataset(
        self,
        epochs: np.ndarray,
        channel_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from all epochs in a dataset.
        
        Args:
            epochs: shape (n_epochs, n_channels, n_samples)
            channel_names: list of channel names
            
        Returns:
            (feature_matrix, feature_names) — shape (n_epochs, n_features)
        """
        feature_dicts = []
        
        for i in range(epochs.shape[0]):
            if i % 50 == 0:
                logger.info(f"Extracting features: epoch {i+1}/{epochs.shape[0]}")
            
            features = self.extract_single_epoch(epochs[i], channel_names)
            feature_dicts.append(features)
        
        # Convert to matrix
        if not feature_dicts:
            return np.array([]), []
        
        self.feature_names = sorted(feature_dicts[0].keys())
        feature_matrix = np.zeros((len(feature_dicts), len(self.feature_names)))
        
        for i, fd in enumerate(feature_dicts):
            for j, name in enumerate(self.feature_names):
                val = fd.get(name, 0.0)
                feature_matrix[i, j] = val if np.isfinite(val) else 0.0
        
        # Replace NaN/Inf
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features from {feature_matrix.shape[0]} epochs")
        
        return feature_matrix, self.feature_names
    
    def fit_selector(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """
        Fit feature selection: variance threshold then SelectKBest.
        
        Args:
            X: feature matrix (n_samples, n_features)
            y: labels (n_samples,)
            feature_names: list of feature names
        """
        self.feature_names = feature_names
        
        # Step 1: Variance threshold
        self.var_selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = self.var_selector.fit_transform(X)
        var_mask = self.var_selector.get_support()
        names_after_var = [n for n, m in zip(feature_names, var_mask) if m]
        
        # Step 2: SelectKBest
        k = min(self.select_k, X_var.shape[1])
        self.kbest_selector = SelectKBest(f_classif, k=k)
        self.kbest_selector.fit(X_var, y)
        kbest_mask = self.kbest_selector.get_support()
        self.selected_feature_names = [n for n, m in zip(names_after_var, kbest_mask) if m]
        
        self._is_fitted = True
        
        logger.info(
            f"Feature selection: {len(feature_names)} → "
            f"{X_var.shape[1]} (variance) → "
            f"{len(self.selected_feature_names)} (SelectKBest)"
        )
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted feature selection to new data."""
        if not self._is_fitted:
            raise RuntimeError("Feature selector not fitted. Call fit_selector first.")
        
        X_var = self.var_selector.transform(X)
        X_selected = self.kbest_selector.transform(X_var)
        
        return X_selected
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit_selector(X, y, feature_names)
        return self.transform(X)
    
    def get_selected_feature_names(self) -> List[str]:
        """Return names of selected features."""
        if self.selected_feature_names is None:
            return self.feature_names or []
        return self.selected_feature_names
