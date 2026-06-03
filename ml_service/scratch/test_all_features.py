import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
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

# Let's extract all columns starting with AB or COH
feature_cols = [col for col in df_filtered.columns if col.startswith('AB.') or col.startswith('COH.')]
print(f"Total clinical EEG features extracted: {len(feature_cols)}")

X = df_filtered[feature_cols].values
y = df_filtered['label'].values

# Handle any NaN values in raw data (if any)
if np.any(np.isnan(X)):
    print("Found NaN values, imputing with column means...")
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

# Filter outliers (Z-score clamping)
mean_val = np.mean(X, axis=0)
std_val = np.std(X, axis=0)
std_val[std_val < 1e-8] = 1.0
z_scores = (X - mean_val) / std_val
outliers = np.abs(z_scores) > 3.0
for row_idx, col_idx in zip(*np.where(outliers)):
    X[row_idx, col_idx] = mean_val[col_idx] + np.sign(z_scores[row_idx, col_idx]) * 3.0 * std_val[col_idx]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("--- Testing different feature selection methods and base classifiers ---")

# Try SelectKBest with f_classif
for k_features in [10, 20, 30, 50, 75, 100, 150, 200, 300]:
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_tr_sel = selector.fit_transform(X_train_s, y_train)
    X_ts_sel = selector.transform(X_test_s)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=42)
    
    stack = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1
    )
    stack.fit(X_tr_sel, y_train)
    acc = stack.score(X_ts_sel, y_test)
    print(f"[SelectKBest f_classif k={k_features}] Stacking accuracy: {acc:.4f}")

# Try Tree-based feature selection (ExtraTrees)
for max_feat in [20, 50, 100, 150]:
    et = ExtraTreesClassifier(n_estimators=100, random_state=42)
    et.fit(X_train_s, y_train)
    importances = et.feature_importances_
    indices = np.argsort(importances)[::-1][:max_feat]
    
    X_tr_sel = X_train_s[:, indices]
    X_ts_sel = X_test_s[:, indices]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, random_state=42)
    
    stack = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1
    )
    stack.fit(X_tr_sel, y_train)
    acc = stack.score(X_ts_sel, y_test)
    print(f"[ExtraTrees selection k={max_feat}] Stacking accuracy: {acc:.4f}")
