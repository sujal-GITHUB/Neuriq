"""
Push accuracy above 60% with advanced techniques:
1. Engineered features (band ratios, asymmetries, mean power per region)
2. More aggressive regularization in ensemble 
3. Broader class scheme (all 7 disorders) for feature selection
4. Repeated stratified k-fold for stability
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               StackingClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
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

# ─── Parse structured EEG feature columns ───
channels = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
bands_6 = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']

# 1. Raw AB features (all 6 bands × 19 channels = 114)
ab_cols = [c for c in df_filtered.columns if c.startswith('AB.')]
print(f"AB columns: {len(ab_cols)}")

# Get AB values as a dict for engineered features
ab_data = {}
for col in ab_cols:
    parts = col.split('.')
    band = parts[2]  # e.g., 'delta'
    ch = parts[4]    # e.g., 'FP1'
    ab_data[(band, ch)] = df_filtered[col].values.astype(float)

# 2. Engineered features from AB data
eng_features = {}

# 2a. Relative Power (per channel, per band)
for ch in channels:
    total = sum(ab_data.get((b, ch), np.zeros(len(df_filtered))) for b in bands_6)
    total = np.maximum(total, 1e-9)
    for b in bands_6:
        val = ab_data.get((b, ch), np.zeros(len(df_filtered)))
        eng_features[f'RP_{b}_{ch}'] = val / total

# 2b. Band ratios (clinically significant)
for ch in channels:
    theta = ab_data.get(('theta', ch), np.zeros(len(df_filtered)))
    alpha = ab_data.get(('alpha', ch), np.zeros(len(df_filtered)))
    beta = ab_data.get(('beta', ch), np.zeros(len(df_filtered)))
    delta = ab_data.get(('delta', ch), np.zeros(len(df_filtered)))
    
    eng_features[f'TBR_{ch}'] = theta / (beta + 1e-9)  # Theta/Beta ratio
    eng_features[f'TAR_{ch}'] = theta / (alpha + 1e-9)  # Theta/Alpha ratio
    eng_features[f'BAR_{ch}'] = beta / (alpha + 1e-9)   # Beta/Alpha ratio
    eng_features[f'DAR_{ch}'] = delta / (alpha + 1e-9)   # Delta/Alpha ratio

# 2c. Hemispheric asymmetry (log ratio)
hom_pairs = [('FP1','FP2'), ('F3','F4'), ('F7','F8'), ('C3','C4'), 
             ('P3','P4'), ('T3','T4'), ('T5','T6'), ('O1','O2')]
for b in bands_6:
    for left, right in hom_pairs:
        l_val = ab_data.get((b, left), np.zeros(len(df_filtered)))
        r_val = ab_data.get((b, right), np.zeros(len(df_filtered)))
        eng_features[f'ASYM_{b}_{left}_{right}'] = np.log((r_val + 1e-9) / (l_val + 1e-9))

# 2d. Regional averages (frontal, central, temporal, parietal, occipital)
regions = {
    'frontal': ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'central': ['C3', 'C4', 'Cz'],
    'temporal': ['T3', 'T4', 'T5', 'T6'],
    'parietal': ['P3', 'P4', 'Pz'],
    'occipital': ['O1', 'O2']
}
for b in bands_6:
    for reg_name, reg_chs in regions.items():
        vals = [ab_data.get((b, ch), np.zeros(len(df_filtered))) for ch in reg_chs]
        eng_features[f'REG_{b}_{reg_name}_mean'] = np.mean(vals, axis=0)
        eng_features[f'REG_{b}_{reg_name}_std'] = np.std(vals, axis=0)

# 2e. Global power per band
for b in bands_6:
    vals = [ab_data.get((b, ch), np.zeros(len(df_filtered))) for ch in channels]
    eng_features[f'GLOBAL_{b}_mean'] = np.mean(vals, axis=0)
    eng_features[f'GLOBAL_{b}_std'] = np.std(vals, axis=0)

# 3. COH features (raw)
coh_cols = [c for c in df_filtered.columns if c.startswith('COH.')]
print(f"COH columns: {len(coh_cols)}")

# Build feature matrix
# Option A: All raw (AB + COH) + engineered
X_raw = df_filtered[ab_cols + coh_cols].values.astype(float)
X_eng = np.column_stack([eng_features[k] for k in sorted(eng_features.keys())])
eng_names = sorted(eng_features.keys())

# Option B: Engineered only
# Option C: Raw + Engineered

# Demographics
df_filtered['sex_int'] = df_filtered['sex'].map({'M': 1, 'F': 0}).fillna(0)
X_demo = df_filtered[['age', 'sex_int']].values.astype(float)

print(f"Engineered features: {X_eng.shape[1]}")
print(f"Raw EEG features: {X_raw.shape[1]}")

# Handle NaNs
def clean(X):
    col_means = np.nanmean(X, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_means, inds[1])
    # Z-score clamp
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    s[s < 1e-8] = 1.0
    z = (X - m) / s
    mask = np.abs(z) > 3.0
    X[mask] = (m + np.sign(z) * 3.0 * s)[mask]
    return X

y = df_filtered['label'].values

configs = {
    'Engineered Only': np.hstack([clean(X_eng.copy()), X_demo]),
    'Raw AB+COH': np.hstack([clean(X_raw.copy()), X_demo]),
    'Raw + Engineered': np.hstack([clean(X_raw.copy()), clean(X_eng.copy()), X_demo]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for config_name, X_config in configs.items():
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name} (shape: {X_config.shape})")
    print(f"{'='*70}")
    
    for k in [20, 30, 40, 50, 75, 100]:
        if k > X_config.shape[1]:
            continue
            
        scores = []
        for train_idx, val_idx in skf.split(X_config, y):
            X_tr, X_val = X_config[train_idx], X_config[val_idx]
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
        
        avg = np.mean(scores)
        marker = " ✓✓✓" if avg >= 0.60 else (" ✓" if avg >= 0.55 else "")
        print(f"  k={k:<3} | 5-Fold CV: {avg:.4f} ± {np.std(scores):.4f}{marker}")

# ─── Also test with just engineered features and more models ───
print(f"\n{'='*70}")
print("BEST CONFIG DEEP DIVE: Engineered features with various models")
print(f"{'='*70}")

X_best = configs['Engineered Only']
for k in [20, 25, 30, 35, 40]:
    for name, model_fn in [
        ("Stacking(3-model)", lambda: StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)),
            ],
            final_estimator=LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
            n_jobs=-1
        )),
        ("Stacking(4-model)", lambda: StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=300, max_depth=7, class_weight='balanced', min_samples_split=5, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=300, max_depth=7, class_weight='balanced', min_samples_split=3, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)),
            ],
            final_estimator=LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced', random_state=42),
            n_jobs=-1, cv=5
        )),
        ("SVM_linear", lambda: SVC(kernel='linear', C=0.1, class_weight='balanced', random_state=42)),
    ]:
        scores = []
        for train_idx, val_idx in skf.split(X_best, y):
            X_tr, X_val = X_best[train_idx], X_best[val_idx]
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
        
        avg = np.mean(scores)
        marker = " ✓✓✓" if avg >= 0.60 else (" ✓" if avg >= 0.55 else "")
        print(f"  k={k:<3} {name:<20} | 5-Fold CV: {avg:.4f} ± {np.std(scores):.4f}{marker}")
