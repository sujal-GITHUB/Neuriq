"""
NeuroAnxiety — Classical ML Models
===================================
SVM, Random Forest, XGBoost, KNN, LDA.
Each model is trained independently on extracted features.

Reference: Gandhi & Jaliya (2024) — hybrid features with meta-model classifier
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available")

from config import (
    SVM_C, SVM_GAMMA, RF_N_ESTIMATORS, RF_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    KNN_N_NEIGHBORS, KNN_METRIC, NUM_CLASSES
)


class ClassicalModels:
    """Container for all classical ML models."""
    
    def __init__(self, n_classes: int = NUM_CLASSES):
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        
        self.models = {
            'svm': SVC(
                C=SVM_C, kernel='rbf', gamma=SVM_GAMMA,
                probability=True, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                random_state=42, n_jobs=-1
            ),
            'knn': KNeighborsClassifier(
                n_neighbors=KNN_N_NEIGHBORS, metric=KNN_METRIC,
                n_jobs=-1
            ),
            'lda': LinearDiscriminantAnalysis()
        }
        
        if HAS_XGBOOST:
            self.models['xgboost'] = XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
                n_jobs=-1
            )
        
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all classical models."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_scaled, y)
            logger.info(f"  {name} trained successfully")
        
        self._is_fitted = True
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        if not self._is_fitted:
            raise RuntimeError("Models not fitted")
        
        X_scaled = self.scaler.transform(X)
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get probability predictions from all models."""
        if not self._is_fitted:
            raise RuntimeError("Models not fitted")
        
        X_scaled = self.scaler.transform(X)
        probas = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probas[name] = model.predict_proba(X_scaled)
            else:
                # Fallback: one-hot from predictions
                preds = model.predict(X_scaled)
                one_hot = np.zeros((len(preds), self.n_classes))
                for i, p in enumerate(preds):
                    one_hot[i, int(p)] = 1.0
                probas[name] = one_hot
        
        return probas
    
    def get_out_of_fold_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate out-of-fold predictions for meta-classifier training.
        
        Returns:
            (oof_predictions, y) — oof shape: (n_samples, n_models * n_classes)
        """
        from sklearn.model_selection import StratifiedKFold
        
        X_scaled = self.scaler.fit_transform(X)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        n_models = len(self.models)
        oof = np.zeros((len(y), n_models * self.n_classes))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train = y[train_idx]
            
            for model_idx, (name, model) in enumerate(self.models.items()):
                # Clone and fit on fold
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                
                # Get probabilities for validation set
                if hasattr(fold_model, 'predict_proba'):
                    proba = fold_model.predict_proba(X_val)
                else:
                    preds = fold_model.predict(X_val)
                    proba = np.zeros((len(preds), self.n_classes))
                    for i, p in enumerate(preds):
                        proba[i, int(p)] = 1.0
                
                start_col = model_idx * self.n_classes
                end_col = start_col + self.n_classes
                oof[val_idx, start_col:end_col] = proba
        
        # Fit all models on full data
        self.fit(X, y)
        
        return oof, y
    
    def save(self, path: str):
        """Save all models and scaler."""
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'n_classes': self.n_classes
        }, path)
        logger.info(f"Classical models saved to {path}")
    
    def load(self, path: str):
        """Load all models and scaler."""
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.n_classes = data['n_classes']
        self._is_fitted = True
        logger.info(f"Classical models loaded from {path}")
