"""
NeuroAnxiety — Comprehensive Evaluation Metrics
================================================
Computes all required metrics for model evaluation.

Per-class: Precision, Recall, F1, Support, Specificity
Overall: Accuracy, Macro/Weighted F1, Cohen's Kappa, MCC, ROC-AUC, AP, Balanced Accuracy
Statistical: McNemar's test, Friedman test
Cross-validation: Mean ± std, training/inference times
"""

import numpy as np
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Compute all evaluation metrics."""
    
    def __init__(self, class_names: List[str] = None, n_classes: int = 3):
        self.class_names = class_names or ['Low', 'Moderate', 'High']
        self.n_classes = n_classes
        self.classes = list(range(n_classes))
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute all metrics.
        
        Args:
            y_true: ground truth labels
            y_pred: predicted labels
            y_proba: predicted probabilities (n_samples, n_classes)
            
        Returns:
            Comprehensive metrics dict
        """
        metrics = {}
        
        # ── Per-Class Metrics ────────────────
        metrics['per_class'] = self._per_class_metrics(y_true, y_pred)
        
        # ── Overall Metrics ──────────────────
        metrics['overall'] = self._overall_metrics(y_true, y_pred, y_proba)
        
        # ── Confusion Matrix ─────────────────
        metrics['confusion_matrix'] = self._confusion_matrix(y_true, y_pred)
        
        # ── ROC Curves ───────────────────────
        if y_proba is not None:
            metrics['roc_curves'] = self._roc_curves(y_true, y_proba)
        
        return metrics
    
    def _per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Per-class precision, recall, F1, support, specificity."""
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        # Support
        support = {}
        for i, name in enumerate(self.class_names):
            support[name] = int(np.sum(y_true == i))
        
        # Specificity (TNR) per class
        cm = confusion_matrix(y_true, y_pred, labels=self.classes)
        specificity = {}
        for i, name in enumerate(self.class_names):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specificity[name] = round(float(tn / max(tn + fp, 1)), 4)
        
        return {
            'precision': {name: round(float(precision[i]), 4) for i, name in enumerate(self.class_names) if i < len(precision)},
            'recall': {name: round(float(recall[i]), 4) for i, name in enumerate(self.class_names) if i < len(recall)},
            'f1_score': {name: round(float(f1[i]), 4) for i, name in enumerate(self.class_names) if i < len(f1)},
            'support': support,
            'specificity': specificity
        }
    
    def _overall_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Overall aggregate metrics."""
        overall = {
            'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'macro_f1': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
            'weighted_f1': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
            'cohens_kappa': round(float(cohen_kappa_score(y_true, y_pred)), 4),
            'mcc': round(float(matthews_corrcoef(y_true, y_pred)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y_true, y_pred)), 4)
        }
        
        # ROC-AUC and AP (require probabilities)
        if y_proba is not None:
            try:
                y_true_bin = label_binarize(y_true, classes=self.classes)
                
                # Per-class ROC-AUC
                roc_auc = {}
                ap = {}
                for i, name in enumerate(self.class_names):
                    if np.sum(y_true_bin[:, i]) > 0:
                        roc_auc[name] = round(float(roc_auc_score(y_true_bin[:, i], y_proba[:, i])), 4)
                        ap[name] = round(float(average_precision_score(y_true_bin[:, i], y_proba[:, i])), 4)
                
                # Macro ROC-AUC
                roc_auc['macro'] = round(float(roc_auc_score(y_true_bin, y_proba, average='macro')), 4)
                
                overall['roc_auc'] = roc_auc
                overall['average_precision'] = ap
                
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
        
        return overall
    
    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Raw and normalized confusion matrices."""
        cm_raw = confusion_matrix(y_true, y_pred, labels=self.classes)
        
        # Normalize per row
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_norm = cm_raw / row_sums
        
        return {
            'raw': cm_raw.tolist(),
            'normalized': np.round(cm_norm, 4).tolist(),
            'labels': self.class_names
        }
    
    def _roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Dict]:
        """Compute ROC curve data per class."""
        y_true_bin = label_binarize(y_true, classes=self.classes)
        curves = {}
        
        for i, name in enumerate(self.class_names):
            if np.sum(y_true_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                # Subsample for JSON-friendly output
                indices = np.linspace(0, len(fpr) - 1, min(50, len(fpr)), dtype=int)
                curves[name] = {
                    'fpr': np.round(fpr[indices], 4).tolist(),
                    'tpr': np.round(tpr[indices], 4).tolist()
                }
        
        return curves
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, float]:
        """McNemar's test between two models."""
        try:
            from statsmodels.stats.contingency_tables import mcnemar
            
            correct1 = (y_pred1 == y_true)
            correct2 = (y_pred2 == y_true)
            
            # Contingency table
            b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
            c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
            
            table = np.array([[np.sum(correct1 & correct2), b], [c, np.sum(~correct1 & ~correct2)]])
            result = mcnemar(table, exact=False, correction=True)
            
            return {
                'statistic': round(float(result.statistic), 4),
                'p_value': round(float(result.pvalue), 6)
            }
        except Exception as e:
            logger.warning(f"McNemar's test failed: {e}")
            return {'statistic': 0.0, 'p_value': 1.0}
    
    @staticmethod
    def friedman_test(fold_accuracies: Dict[str, List[float]]) -> Dict[str, float]:
        """Friedman test across model accuracies per fold."""
        try:
            from scipy.stats import friedmanchisquare
            
            arrays = list(fold_accuracies.values())
            if len(arrays) < 3:
                return {'statistic': 0.0, 'p_value': 1.0}
            
            stat, p_value = friedmanchisquare(*arrays)
            
            return {
                'statistic': round(float(stat), 4),
                'p_value': round(float(p_value), 6)
            }
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
            return {'statistic': 0.0, 'p_value': 1.0}
    
    @staticmethod
    def cross_validation_summary(
        fold_metrics: List[Dict[str, float]],
        metric_name: str = 'accuracy'
    ) -> Dict[str, float]:
        """Compute mean ± std for a metric across folds."""
        values = [fm.get(metric_name, 0.0) for fm in fold_metrics]
        return {
            'mean': round(float(np.mean(values)), 4),
            'std': round(float(np.std(values)), 4),
            'min': round(float(np.min(values)), 4),
            'max': round(float(np.max(values)), 4),
            'values': [round(float(v), 4) for v in values]
        }
    
    @staticmethod
    def measure_inference_time(model, X_sample: np.ndarray, n_runs: int = 100) -> float:
        """Measure average inference time in milliseconds."""
        import torch
        
        if isinstance(model, torch.nn.Module):
            model.eval()
            x = torch.FloatTensor(X_sample[:1])
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    model(x)
            
            # Measure
            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    start = time.perf_counter()
                    model(x)
                    times.append((time.perf_counter() - start) * 1000)
            
            return round(float(np.mean(times)), 3)
        else:
            # sklearn model
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                model.predict(X_sample[:1])
                times.append((time.perf_counter() - start) * 1000)
            
            return round(float(np.mean(times)), 3)
