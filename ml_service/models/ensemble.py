"""
NeuroAnxiety — Meta-Classifier Ensemble
========================================
Logistic Regression meta-classifier trained on out-of-fold predictions
from the classical model ensemble (SVM, RF, XGBoost, KNN, LDA).

Reproduces the Gandhi & Jaliya (2024) hybrid approach.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

from models.classical import ClassicalModels
from config import NUM_CLASSES, CV_FOLDS

logger = logging.getLogger(__name__)


class MetaClassifierEnsemble:
    """
    Stacking ensemble:
    Level 0: SVM, RF, XGBoost, KNN, LDA (trained on features)
    Level 1: Logistic Regression (trained on out-of-fold probabilities)
    """
    
    def __init__(self, n_classes: int = NUM_CLASSES, cv_folds: int = CV_FOLDS):
        self.n_classes = n_classes
        self.cv_folds = cv_folds
        self.base_models = ClassicalModels(n_classes)
        self.meta_scaler = StandardScaler()
        self.meta_classifier = LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000,
            multi_class='multinomial', random_state=42
        )
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the full stacking ensemble.
        
        1. Generate out-of-fold predictions from base models
        2. Train meta-classifier on OOF predictions
        3. Re-fit base models on full data
        """
        logger.info("Training meta-classifier ensemble...")
        
        # Step 1: Get out-of-fold predictions
        oof_predictions, _ = self.base_models.get_out_of_fold_predictions(
            X, y, self.cv_folds
        )
        
        # Step 2: Train meta-classifier
        oof_scaled = self.meta_scaler.fit_transform(oof_predictions)
        self.meta_classifier.fit(oof_scaled, y)
        
        self._is_fitted = True
        logger.info("Meta-classifier ensemble training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the full ensemble pipeline."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        # Get base model probabilities
        probas = self.base_models.predict_proba(X)
        
        # Stack probabilities as meta-features
        meta_features = np.hstack([probas[name] for name in sorted(probas.keys())])
        meta_scaled = self.meta_scaler.transform(meta_features)
        
        return self.meta_classifier.predict(meta_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from the ensemble."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        probas = self.base_models.predict_proba(X)
        meta_features = np.hstack([probas[name] for name in sorted(probas.keys())])
        meta_scaled = self.meta_scaler.transform(meta_features)
        
        return self.meta_classifier.predict_proba(meta_scaled)
    
    def get_base_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get individual base model predictions for analysis."""
        return self.base_models.predict(X)
    
    def save(self, path: str):
        """Save the entire ensemble."""
        joblib.dump({
            'base_models': self.base_models,
            'meta_scaler': self.meta_scaler,
            'meta_classifier': self.meta_classifier,
            'n_classes': self.n_classes,
            'cv_folds': self.cv_folds
        }, path)
        logger.info(f"Ensemble saved to {path}")
    
    def load(self, path: str):
        """Load ensemble from file."""
        data = joblib.load(path)
        self.base_models = data['base_models']
        self.meta_scaler = data['meta_scaler']
        self.meta_classifier = data['meta_classifier']
        self.n_classes = data['n_classes']
        self.cv_folds = data['cv_folds']
        self._is_fitted = True
        logger.info(f"Ensemble loaded from {path}")
