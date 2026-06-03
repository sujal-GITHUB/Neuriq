"""
NeuroAnxiety — Multi-Class EEG Psychiatric Disorders Dataset Generator (Step 1)
==============================================================================
Generates a realistic, highly rigorous synthetic dataset representing the 
Kaggle Multi-Class EEG Dataset for Psychiatric Disorders.
Contains 1,200 patient recordings with 95 multi-channel spectral power columns 
representing Delta, Theta, Alpha, Beta, and Gamma bands across 19 standard channels.
Explicitly maps to 3 target categories:
  - Class 0: Healthy Controls (Baseline State)
  - Class 1: Social Anxiety Disorder manifestation
  - Class 2: Acute Stress State
Also injects missing values, duplicate records, and massive outlier spikes to validate 
the Preprocessing Pipeline (Step 2).
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_PATIENTS = 1200
CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
]
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

def generate_eeg_mental_health_dataset():
    """Generates the synthetic multi-class EEG database with engineered feature shifts."""
    records = []
    
    # 95 feature columns (19 channels * 5 bands)
    feature_names = [f"PSD_{band}_{channel}" for channel in CHANNELS for band in BANDS]
    
    # Generate label distribution (balanced: 400 Healthy, 400 Anxiety, 400 Stress)
    labels = np.repeat([0, 1, 2], N_PATIENTS // 3)
    
    for idx, label in enumerate(labels):
        record = {}
        # Patient demographics
        record["patient_id"] = f"SUBJ_{idx:04d}"
        record["age"] = int(np.random.normal(32.0 if label == 0 else (28.0 if label == 1 else 35.0), 8.0))
        record["age"] = max(18, min(70, record["age"]))
        record["sex"] = np.random.choice(["M", "F"], p=[0.48, 0.52])
        
        # Brainwave feature simulation based on literature findings:
        # - Healthy: Normal baseline distributions, alpha dominance in posterior
        # - Social Anxiety: High frontal beta/gamma, low frontal alpha, high theta/alpha ratios
        # - Acute Stress: Widespread high beta, reduced alpha, elevated delta/theta
        for ch in CHANNELS:
            is_frontal = ch in ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
            is_posterior = ch in ['P3', 'P4', 'O1', 'O2', 'Pz']
            
            for band in BANDS:
                col_name = f"PSD_{band}_{ch}"
                
                # Base distribution
                mu = 15.0
                sigma = 3.5
                
                if label == 0:  # Healthy Controls
                    if band == 'Alpha' and is_posterior:
                        mu = 28.0  # Dominant Alpha
                        sigma = 4.0
                    elif band in ['Beta', 'Gamma']:
                        mu = 10.0
                        sigma = 2.0
                        
                elif label == 1:  # Social Anxiety Disorder
                    if band in ['Beta', 'Gamma'] and is_frontal:
                        mu = 29.5  # High frontal high-frequency bands
                        sigma = 5.0
                    elif band == 'Alpha' and is_frontal:
                        mu = 8.5   # Frontal alpha asymmetry / hypoactivity
                        sigma = 1.5
                    elif band == 'Theta':
                        mu = 22.0
                        sigma = 3.5
                        
                elif label == 2:  # Acute Stress State
                    if band == 'Beta':
                        mu = 32.0  # Widespread hyper-beta activity
                        sigma = 5.5
                    elif band == 'Alpha':
                        mu = 9.0   # Blocked alpha
                        sigma = 2.0
                    elif band == 'Delta':
                        mu = 24.0  # Slow wave stress signature
                        sigma = 4.0
                
                val = np.random.normal(mu, sigma)
                record[col_name] = round(max(0.1, val), 4)
                
        record["label"] = label
        records.append(record)
        
    df = pd.DataFrame(records)
    
    # ── Inject Preprocessing Challenges (Step 2) ──────────────────────────
    # 1. Inject brief sensor dropouts / missing values (NaNs) in ~5% of cells
    mask_missing = np.random.random(size=(N_PATIENTS, len(feature_names))) < 0.03
    for col_idx, col_name in enumerate(feature_names):
        df.loc[mask_missing[:, col_idx], col_name] = np.nan
        
    # 2. Inject massive outlier spikes (Z-score > 3) to test clamping
    # We will choose 50 random cells and inject extreme values (e.g. 150.0 µV)
    for _ in range(50):
        r = np.random.randint(0, N_PATIENTS)
        c = np.random.choice(feature_names)
        df.loc[r, c] = np.random.choice([180.0, 220.0])
        
    # 3. Inject duplicate rows (to test prediction bias duplicate removal)
    # Duplicate 15 random rows
    dup_rows = df.sample(n=15, random_state=42)
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

if __name__ == "__main__":
    os.makedirs("ml_service/datasets", exist_ok=True)
    df = generate_eeg_mental_health_dataset()
    df.to_csv("ml_service/datasets/eeg_mental_health.csv", index=False)
    print(f"Generated mental health dataset with shape: {df.shape}")
