"""
NeuroAnxiety — PCA-RFE Hybrid Feature Extraction Pipeline (Step 3)
==================================================================
Implements the core mathematical innovation of the paper:
  Phase 1: Principal Component Analysis (PCA)
    Transforms the highly dimensional, correlated electrode frequency bands
    into entirely uncorrelated Principal Components (PCs) representing maximal dataset variance.
  Phase 2: Recursive Feature Elimination (RFE)
    Runs RFE directly in the PCA-transformed dataset space using a tree-based estimator
    to select only the most powerful components (retaining the best predictive indices).
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List, Optional

class PCARFEHybridSelector:
    """Combines Principal Component Analysis and Recursive Feature Elimination."""
    
    def __init__(
        self,
        pca_components: int = 95,
        rfe_select_components: int = 10,
        random_state: int = 42
    ):
        self.pca_components = pca_components
        self.rfe_select_components = rfe_select_components
        self.random_state = random_state
        
        # PCA Model
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        
        # RFE Tree-Based Estimator
        self.estimator = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state
        )
        self.rfe = RFE(
            estimator=self.estimator,
            n_features_to_select=self.rfe_select_components,
            step=1
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PCARFEHybridSelector':
        """Fits PCA first, then fits RFE on the PCA component space."""
        # Clean inputs
        X = np.nan_to_num(X)
        
        # 1. Fit PCA
        n_comp = min(self.pca_components, X.shape[1])
        if n_comp != self.pca.n_components:
            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
            
        X_pca = self.pca.fit_transform(X)
        
        # 2. Fit RFE on top of PCA components
        n_sel = min(self.rfe_select_components, X_pca.shape[1])
        if n_sel != self.rfe.n_features_to_select:
            self.rfe = RFE(
                estimator=self.estimator,
                n_features_to_select=n_sel,
                step=1
            )
            
        self.rfe.fit(X_pca, y)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms features through PCA, then filters using RFE support mask."""
        X = np.nan_to_num(X)
        X_pca = self.pca.transform(X)
        X_selected = self.rfe.transform(X_pca)
        return X_selected
        
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Helper to fit and transform in one step."""
        return self.fit(X, y).transform(X)
        
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Returns the variance ratio explained by individual components."""
        return self.pca.explained_variance_ratio_
        
    def get_selected_components_indices(self) -> List[int]:
        """Returns the 0-indexed indices of the PCA components selected by RFE."""
        support = self.rfe.support_
        return [i for i, val in enumerate(support) if val]
