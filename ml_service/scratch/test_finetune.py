"""Fine-tune around the sweet-spot found in the broad sweep:
  - SelectKBest k=30-60 with all AB+COH features
  - SMOTE oversampling to balance the 3 classes (128/107/95)
  - Wider hyperparameter grid for the Stacking ensemble
  - Additional base models (GradientBoosting, SVM)
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               StackingClassifier, GradientBoostingClassifier,
                               BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
    print("SMOTE available ✓")
except ImportError:
    HAS_SMOTE = False
    print("SMOTE not available, skipping oversampling")

data_path = "/home/s/Work/Neuriq/ml_service/datasets/EEG.machinelearing_data_BRMH.csv"
df_real = pd.read_csv(data_path)

target_classes = {
    'Healthy control': 0,
    'Anxiety disorder': 1,
    'Trauma and stress related disorder': 2
}
df_filtered = df_real[df_real['main.disorder'].isin(target_classes.keys())].copy()
df_filtered['label'] = df_filtered['main.disorder'].map(target_classes)

# All EEG features
feature_cols = [col for col in df_filtered.columns if col.startswith('AB.') or col.startswith('COH.')]
X = df_filtered[feature_cols].values
y = df_filtered['label'].values

# Handle NaNs
if np.any(np.isnan(X)):
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

# Z-score outlier clamping
mean_val = np.mean(X, axis=0)
std_val = np.std(X, axis=0)
std_val[std_val < 1e-8] = 1.0
z_scores = (X - mean_val) / std_val
outliers = np.abs(z_scores) > 3.0
X[outliers] = (mean_val + np.sign(z_scores) * 3.0 * std_val)[outliers]

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, 3 classes")

# Stratified K-Fold 
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(X_data, y_data, k_feat, model_fn, use_smote=False, scoring_func=f_classif):
    scores = []
    for train_idx, val_idx in skf.split(X_data, y_data):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        selector = SelectKBest(score_func=scoring_func, k=k_feat)
        X_tr_s = selector.fit_transform(X_tr_s, y_tr)
        X_val_s = selector.transform(X_val_s)
        
        if use_smote and HAS_SMOTE:
            smote = SMOTE(random_state=42)
            X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)
        
        clf = model_fn()
        clf.fit(X_tr_s, y_tr)
        scores.append(accuracy_score(y_val, clf.predict(X_val_s)))
    return np.mean(scores)

# ===== EXPERIMENT 1: Fine-tune k around the sweet spot =====
print("\n=== Experiment 1: Fine-grained k sweep with Stacking ===")
for k in [20, 25, 30, 35, 40, 45, 50, 55, 60]:
    def make_stacking():
        return StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_jobs=-1
        )
    acc = evaluate(X, y, k, make_stacking)
    print(f"  k={k:<3} | Stacking CV Accuracy: {acc:.4f}")

# ===== EXPERIMENT 2: SMOTE oversampling =====
print("\n=== Experiment 2: SMOTE oversampling with Stacking ===")
for k in [25, 30, 35, 40, 45, 50, 55]:
    def make_stacking():
        return StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_jobs=-1
        )
    acc = evaluate(X, y, k, make_stacking, use_smote=True)
    print(f"  k={k:<3} SMOTE | Stacking CV Accuracy: {acc:.4f}")

# ===== EXPERIMENT 3: Stronger base models =====
print("\n=== Experiment 3: Stronger base models ===")
for k in [30, 40, 50]:
    def make_strong_stacking():
        return StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=7, min_samples_split=5, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=300, max_depth=7, min_samples_split=3, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)),
            ],
            final_estimator=LogisticRegression(C=0.5, max_iter=2000, random_state=42),
            n_jobs=-1, cv=5
        )
    acc = evaluate(X, y, k, make_strong_stacking)
    print(f"  k={k:<3} | Strong Stacking CV Accuracy: {acc:.4f}")
    
    acc_smote = evaluate(X, y, k, make_strong_stacking, use_smote=True)
    print(f"  k={k:<3} SMOTE | Strong Stacking CV Accuracy: {acc_smote:.4f}")

# ===== EXPERIMENT 4: mutual_info feature scoring =====
print("\n=== Experiment 4: mutual_info_classif scoring ===")
for k in [30, 40, 50]:
    def make_stacking():
        return StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=150, max_depth=5, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_jobs=-1
        )
    acc = evaluate(X, y, k, make_stacking, scoring_func=mutual_info_classif)
    print(f"  k={k:<3} MI | Stacking CV Accuracy: {acc:.4f}")

# ===== EXPERIMENT 5: SVM with careful tuning =====
print("\n=== Experiment 5: Tuned SVM ===")
for k in [15, 20, 25, 30]:
    for C in [0.01, 0.1, 1.0]:
        def make_svm(C=C):
            return SVC(kernel='linear', C=C, random_state=42)
        acc = evaluate(X, y, k, make_svm)
        print(f"  k={k:<3} C={C:<5} | SVM Linear CV Accuracy: {acc:.4f}")
