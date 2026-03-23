"""
Neuriq — Multi-Boosting Ensemble
====================================
A gradient boosted ensemble incorporating XGBoost, LightGBM, and CatBoost.
Soft voting is used to average their predicted probabilities.
Hyperparameters can be tuned via Optuna.

Reported to achieve 97.33% accuracy in a 2024 study on anxiety EEG classification.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib

from config import NUM_CLASSES, CV_FOLDS

logger = logging.getLogger(__name__)


class BoostingEnsemble:
    """
    Ensemble of XGBoost, LightGBM, and CatBoost.
    """
    
    def __init__(self, n_classes: int = NUM_CLASSES):
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.models = {}
        self._is_fitted = False
        
        if not all([XGBClassifier, LGBMClassifier, CatBoostClassifier]):
            logger.warning("One or more boosting libraries are missing. Check requirements.")
    
    def fit(self, X: np.ndarray, y: np.ndarray, tune_hyperparams: bool = False):
        """Train the boosting ensemble."""
        logger.info("Training Boosting Ensemble (XGB, LightGBM, CatBoost)...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Default parameters
        xgb_params = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'eval_metric': 'mlogloss', 'random_state': 42}
        lgb_params = {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1}
        cat_params = {'iterations': 300, 'depth': 6, 'learning_rate': 0.05, 'loss_function': 'MultiClass', 'random_seed': 42, 'verbose': False}
        
        if tune_hyperparams:
            logger.info("Tuning hyperparameters with Optuna...")
            params = self._tune_with_optuna(X_scaled, y)
            xgb_params.update(params['xgb'])
            lgb_params.update(params['lgb'])
            cat_params.update(params['cat'])
        
        self.models = {
            'xgboost': XGBClassifier(**xgb_params),
            'lightgbm': LGBMClassifier(**lgb_params),
            'catboost': CatBoostClassifier(**cat_params)
        }
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_scaled, y)
        
        self._is_fitted = True
        logger.info("Boosting ensemble training complete.")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Soft voting prediction probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble is not fitted.")
            
        X_scaled = self.scaler.transform(X)
        probas = []
        
        for name, model in self.models.items():
            probas.append(model.predict_proba(X_scaled))
            
        # Average the probabilities
        mean_proba = np.mean(probas, axis=0)
        return mean_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def _tune_with_optuna(self, X: np.ndarray, y: np.ndarray, n_trials: int = 15) -> Dict[str, Dict]:
        """Use Optuna to find best hyperparameters."""
        def objective(trial):
            # Sample parameters
            xgb_depth = trial.suggest_int('xgb_depth', 3, 9)
            lgb_depth = trial.suggest_int('lgb_depth', 3, 9)
            cat_depth = trial.suggest_int('cat_depth', 3, 9)
            
            lr = trial.suggest_float('lr', 0.01, 0.2, log=True)
            n_est = trial.suggest_int('n_estimators', 100, 500, step=100)
            
            xgb = XGBClassifier(max_depth=xgb_depth, learning_rate=lr, n_estimators=n_est, random_state=42, eval_metric='mlogloss')
            lgb = LGBMClassifier(max_depth=lgb_depth, learning_rate=lr, n_estimators=n_est, random_state=42, verbose=-1)
            cat = CatBoostClassifier(depth=cat_depth, learning_rate=lr, iterations=n_est, random_seed=42, verbose=False, loss_function='MultiClass')
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_accs = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]
                
                xgb.fit(X_tr, y_tr)
                lgb.fit(X_tr, y_tr)
                cat.fit(X_tr, y_tr)
                
                p_xgb = xgb.predict_proba(X_va)
                p_lgb = lgb.predict_proba(X_va)
                p_cat = cat.predict_proba(X_va)
                
                p_ensemble = (p_xgb + p_lgb + p_cat) / 3.0
                preds = np.argmax(p_ensemble, axis=1)
                
                fold_accs.append(accuracy_score(y_va, preds))
                
            return np.mean(fold_accs)
            
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best = study.best_params
        
        return {
            'xgb': {'max_depth': best['xgb_depth'], 'learning_rate': best['lr'], 'n_estimators': best['n_estimators']},
            'lgb': {'max_depth': best['lgb_depth'], 'learning_rate': best['lr'], 'n_estimators': best['n_estimators']},
            'cat': {'depth': best['cat_depth'], 'learning_rate': best['lr'], 'iterations': best['n_estimators']}
        }

    def save(self, path: str):
        joblib.dump({
            'models': self.models,
            'scaler': self.scaler,
            'n_classes': self.n_classes
        }, path)
        logger.info(f"Boosting Ensemble saved to {path}")
    
    def load(self, path: str):
        data = joblib.load(path)
        self.models = data['models']
        self.scaler = data['scaler']
        self.n_classes = data['n_classes']
        self._is_fitted = True
        logger.info(f"Boosting Ensemble loaded from {path}")
