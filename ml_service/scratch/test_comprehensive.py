"""
Comprehensive accuracy improvement for BRMH clinical EEG dataset.
Key improvements:
1. Include ALL AB bands (delta, theta, alpha, beta, highbeta, gamma) = 114 features
2. Include COH (coherence) features = 1026 features  
3. Add demographics (age, sex)
4. Use BOTH train/test accuracy AND 5-fold CV for validation
5. Try class-weighted models to handle imbalance
6. Explore broader hyperparameter space
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               StackingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

data_path = "/home/s/Work/Neuriq/ml_service/datasets/EEG.machinelearing_data_BRMH.csv"
df_real = pd.read_csv(data_path)

target_classes = {
    'Healthy control': 0,
    'Anxiety disorder': 1,
    'Trauma and stress related disorder': 2
}
df_filtered = df_real[df_real['main.disorder'].isin(target_classes.keys())].copy()
df_filtered['label'] = df_filtered['main.disorder'].map(target_classes)

# ALL EEG features (AB + COH)
eeg_cols = [col for col in df_filtered.columns if col.startswith('AB.') or col.startswith('COH.')]
print(f"EEG feature columns: {len(eeg_cols)}")

# Add demographics
df_filtered['sex_int'] = df_filtered['sex'].map({'M': 1, 'F': 0}).fillna(0)
demo_cols = ['age', 'sex_int']

all_cols = eeg_cols + demo_cols
X = df_filtered[all_cols].values.astype(float)
y = df_filtered['label'].values

# Handle NaNs
col_means = np.nanmean(X, axis=0)
col_means = np.nan_to_num(col_means, nan=0.0)
inds = np.where(np.isnan(X))
if len(inds[0]) > 0:
    X[inds] = np.take(col_means, inds[1])

# Z-score outlier clamping
mean_v = np.mean(X, axis=0)
std_v = np.std(X, axis=0)
std_v[std_v < 1e-8] = 1.0
z = (X - mean_v) / std_v
mask = np.abs(z) > 3.0
X[mask] = (mean_v + np.sign(z) * 3.0 * std_v)[mask]

print(f"Final X shape: {X.shape}, y shape: {y.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# === Test 1: Single train/test split (same as test_models.py to reproduce 60%) ===
print("\n" + "="*70)
print("TEST 1: Single Train/Test Split (random_state=42)")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

best_acc = 0
best_config = ""

for k in [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 100]:
    selector = SelectKBest(score_func=f_classif, k=k)
    X_tr = selector.fit_transform(X_train_s, y_train)
    X_ts = selector.transform(X_test_s)
    
    # Class-weighted Stacking
    stack = StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, 
                                          class_weight='balanced', random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=6,
                                        class_weight='balanced', random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, 
                                       subsample=0.8, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        n_jobs=-1
    )
    stack.fit(X_tr, y_train)
    acc = stack.score(X_ts, y_test)
    if acc > best_acc:
        best_acc = acc
        best_config = f"Stacking(balanced) k={k}"
    print(f"  Stacking(balanced) k={k:<3} => Test Acc: {acc:.4f}")

    # Simple RF with class_weight
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, 
                                 class_weight='balanced', random_state=42)
    rf.fit(X_tr, y_train)
    acc_rf = rf.score(X_ts, y_test)
    if acc_rf > best_acc:
        best_acc = acc_rf
        best_config = f"RF(balanced) k={k}"
    print(f"  RF(balanced) k={k:<3} => Test Acc: {acc_rf:.4f}")

    # SVM linear
    svm = SVC(kernel='linear', C=0.1, class_weight='balanced', random_state=42)
    svm.fit(X_tr, y_train)
    acc_svm = svm.score(X_ts, y_test)
    if acc_svm > best_acc:
        best_acc = acc_svm
        best_config = f"SVM(linear,balanced) k={k}"
    print(f"  SVM(linear,balanced) k={k:<3} => Test Acc: {acc_svm:.4f}")

print(f"\n*** BEST Single Split: {best_config} => {best_acc:.4f} ***")

# === Test 2: Multiple random seeds for train/test split ===
print("\n" + "="*70)
print("TEST 2: Multiple Random Seeds (robustness check)")
print("="*70)

for k in [30, 40, 50]:
    seed_accs = []
    for seed in [42, 123, 456, 789, 2024, 1337, 7, 99, 314, 555]:
        X_tr_s, X_ts_s, y_tr_s, y_ts_s = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_s)
        X_ts_sc = sc.transform(X_ts_s)
        
        sel = SelectKBest(score_func=f_classif, k=k)
        X_tr_sel = sel.fit_transform(X_tr_sc, y_tr_s)
        X_ts_sel = sel.transform(X_ts_sc)
        
        stack = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                              class_weight='balanced', random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=200, max_depth=6,
                                            class_weight='balanced', random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                                           subsample=0.8, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
            n_jobs=-1
        )
        stack.fit(X_tr_sel, y_tr_s)
        seed_accs.append(stack.score(X_ts_sel, y_ts_s))
    
    avg = np.mean(seed_accs)
    std = np.std(seed_accs)
    print(f"  k={k} | 10-seed avg: {avg:.4f} ± {std:.4f} | range [{min(seed_accs):.4f}, {max(seed_accs):.4f}]")

# === Test 3: 5-Fold CV (gold standard) ===
print("\n" + "="*70)
print("TEST 3: 5-Fold Stratified CV (gold standard)")
print("="*70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in [15, 20, 25, 30, 35, 40, 50, 60, 75, 100]:
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        
        sel = SelectKBest(score_func=f_classif, k=k)
        X_tr_sel = sel.fit_transform(X_tr_s, y_tr)
        X_val_sel = sel.transform(X_val_s)
        
        stack = StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                              class_weight='balanced', random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=200, max_depth=6,
                                            class_weight='balanced', random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4,
                                           subsample=0.8, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
            n_jobs=-1
        )
        stack.fit(X_tr_sel, y_tr)
        scores.append(stack.score(X_val_sel, y_val))
    
    print(f"  k={k:<3} | 5-Fold CV: {np.mean(scores):.4f} ± {np.std(scores):.4f} | folds: {[f'{s:.3f}' for s in scores]}")

# Test with more models at best k values
print("\n--- Additional models at k=50 ---")
k = 50
for name, model_fn in [
    ("GradientBoosting(balanced)", lambda: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
    ("XGB(scale_pos)", lambda: xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)),
    ("RF(300,d8,balanced)", lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)),
    ("SVM(rbf,balanced)", lambda: SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)),
    ("SVM(linear,balanced)", lambda: SVC(kernel='linear', C=0.1, class_weight='balanced', random_state=42)),
]:
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_val_s = sc.transform(X_val)
        
        sel = SelectKBest(score_func=f_classif, k=k)
        X_tr_sel = sel.fit_transform(X_tr_s, y_tr)
        X_val_sel = sel.transform(X_val_s)
        
        clf = model_fn()
        clf.fit(X_tr_sel, y_tr)
        scores.append(clf.score(X_val_sel, y_val))
    
    print(f"  {name:<30} | 5-Fold CV: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
