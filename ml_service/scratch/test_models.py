import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

# Extract all EEG features (AB and COH columns)
feature_cols = [col for col in df_filtered.columns if col.startswith('AB.') or col.startswith('COH.')]
X = df_filtered[feature_cols].values
y = df_filtered['label'].values

# Demographics as additional features
df_filtered['sex_int'] = df_filtered['sex'].map({'M': 1, 'F': 0})
X_demo = df_filtered[['age', 'sex_int']].values
X = np.hstack([X, X_demo])

# Handle NaNs
if np.any(np.isnan(X)):
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

# Outlier clamping
mean_val = np.mean(X, axis=0)
std_val = np.std(X, axis=0)
std_val[std_val < 1e-8] = 1.0
z_scores = (X - mean_val) / std_val
outliers = np.abs(z_scores) > 3.0
for row_idx, col_idx in zip(*np.where(outliers)):
    X[row_idx, col_idx] = mean_val[col_idx] + np.sign(z_scores[row_idx, col_idx]) * 3.0 * std_val[col_idx]

# Define cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_config(X_data, y_data, k_features, model_name, model_fn):
    scores = []
    for train_idx, val_idx in skf.split(X_data, y_data):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_data[train_idx], y_data[val_idx]
        
        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        # Select K Best
        if k_features is not None:
            selector = SelectKBest(score_func=f_classif, k=k_features)
            X_tr_s = selector.fit_transform(X_tr_s, y_tr)
            X_val_s = selector.transform(X_val_s)
            
        clf = model_fn()
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_val_s)
        scores.append(accuracy_score(y_val, y_pred))
        
    return np.mean(scores)

# List of models to try
models = {
    "SVM_RBF": lambda: SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    "SVM_Linear": lambda: SVC(kernel='linear', C=0.1, probability=True, random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=150, max_depth=6, random_state=42),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
    "LogisticRegression": lambda: LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    "Stacking_LR": lambda: StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42))
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1
    )
}

print("Running 5-Fold Stratified Cross-Validation on original dataset...")
results = []
for k_feat in [15, 30, 50, 100, 200, 300, None]:
    for name, fn in models.items():
        score = evaluate_config(X, y, k_feat, name, fn)
        print(f"Features: {k_feat if k_feat else 'All':<5} | Model: {name:<20} | 5-Fold CV Accuracy: {score:.4f}")
        results.append((k_feat, name, score))

# Print top 5 results
results.sort(key=lambda x: x[2], reverse=True)
print("\n--- TOP 5 CONFIGURATIONS ---")
for r in results[:5]:
    print(f"Features: {r[0] if r[0] else 'All':<5} | Model: {r[1]:<20} | CV Accuracy: {r[2]:.4f}")
