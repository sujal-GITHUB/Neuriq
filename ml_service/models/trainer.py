"""
NeuroAnxiety — Training Pipeline
=================================
Handles training for both classical and deep learning models.

Features:
  - 5-fold stratified cross-validation
  - LOSO (Leave-One-Subject-Out) evaluation
  - Adam optimizer with CosineAnnealingLR
  - CrossEntropyLoss with label smoothing
  - Early stopping with patience=15
  - Checkpoint saving
"""

import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from config import (
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, LABEL_SMOOTHING, CV_FOLDS,
    CHECKPOINT_DIR, NUM_CLASSES
)

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, n_classes: int = 3):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = nn.functional.log_softmax(pred, dim=-1)
        
        # One-hot with smoothing
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob, self.smoothing / (self.n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_target * log_prob).sum(dim=-1).mean()
        return loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class Trainer:
    """Unified trainer for deep learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        n_classes: int = NUM_CLASSES,
        batch_size: int = BATCH_SIZE,
        max_epochs: int = MAX_EPOCHS,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        patience: int = EARLY_STOP_PATIENCE,
        label_smoothing: float = LABEL_SMOOTHING,
        device: str = None
    ):
        self.model = model
        self.model_name = model_name
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.criterion = LabelSmoothingCrossEntropy(label_smoothing, n_classes)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        
        self.training_history = []
        self.progress_callback: Optional[Callable] = None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)
        
        return {
            'train_loss': total_loss / total,
            'train_accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        return {
            'val_loss': total_loss / total,
            'val_accuracy': correct / total,
            'predictions': np.array(all_preds),
            'labels': np.array(all_labels)
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.
        
        Args:
            X_train, y_train: training data
            X_val, y_val: validation data
            progress_callback: (epoch, metrics) → None
        """
        self.progress_callback = progress_callback
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        early_stopping = EarlyStopping(patience=self.patience)
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()
            
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            metrics = {
                **train_metrics,
                'val_loss': val_metrics['val_loss'],
                'val_accuracy': val_metrics['val_accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'epoch': epoch
            }
            
            self.training_history.append(metrics)
            
            if progress_callback:
                progress_callback(epoch, metrics)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            # Early stopping
            if early_stopping(val_metrics['val_loss']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{self.max_epochs} - "
                    f"Loss: {train_metrics['train_loss']:.4f}/{val_metrics['val_loss']:.4f} - "
                    f"Acc: {train_metrics['train_accuracy']:.4f}/{val_metrics['val_accuracy']:.4f}"
                )
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Final validation
        final_metrics = self.validate(val_loader)
        
        return {
            'history': self.training_history,
            'best_val_loss': best_val_loss,
            'final_predictions': final_metrics['predictions'],
            'final_labels': final_metrics['labels'],
            'val_accuracy': final_metrics['val_accuracy']
        }
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = CV_FOLDS,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """5-fold stratified cross-validation."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_predictions = np.zeros(len(y), dtype=int)
        all_probabilities = np.zeros((len(y), self.n_classes))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"\n--- Fold {fold + 1}/{n_folds} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Reset model weights
            self.model.apply(self._reset_weights)
            self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)
            
            fold_start = time.time()
            result = self.train(X_train, y_train, X_val, y_val, progress_callback)
            fold_time = time.time() - fold_start
            
            result['fold'] = fold + 1
            result['training_time'] = fold_time
            fold_results.append(result)
            
            all_predictions[val_idx] = result['final_predictions']
        
        return {
            'fold_results': fold_results,
            'all_predictions': all_predictions,
            'all_labels': y,
            'mean_accuracy': np.mean([r['val_accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['val_accuracy'] for r in fold_results])
        }
    
    def loso_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Leave-One-Subject-Out cross-validation."""
        logo = LeaveOneGroupOut()
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, subject_ids)):
            subject = subject_ids[val_idx[0]]
            logger.info(f"\n--- LOSO: Subject {subject} ---")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.apply(self._reset_weights)
            self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)
            
            result = self.train(X_train, y_train, X_val, y_val, progress_callback)
            result['subject'] = int(subject)
            fold_results.append(result)
        
        return {
            'fold_results': fold_results,
            'mean_accuracy': np.mean([r['val_accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['val_accuracy'] for r in fold_results])
        }
    
    def save_checkpoint(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = str(CHECKPOINT_DIR / f"{self.model_name}_best.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_name': self.model_name
        }, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint loaded: {path}")
    
    @staticmethod
    def _reset_weights(module):
        """Reset module weights for fresh training per fold."""
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
