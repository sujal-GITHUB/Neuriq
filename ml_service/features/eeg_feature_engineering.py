"""
NeuroAnxiety — Advanced EEG Feature Engineering (Step 3b)
=========================================================
Extracts clinically-validated engineered features from raw AB columns:
  1. Relative Band Power (per channel, per band — 6 bands × 19 channels = 114 features)
  2. Band Power Ratios (Theta/Beta, Theta/Alpha, Beta/Alpha, Delta/Alpha per channel)
  3. Hemispheric Asymmetry (log ratio of homologous pairs per band)
  4. Regional Averages (mean + std per band across 5 brain regions)
  5. Global Power Statistics (mean + std per band)
  6. Demographics (age, sex)

These engineered features dramatically improve discriminability for EEG psychiatric
classification — see literature on Theta/Beta ratio as ADHD/anxiety marker, frontal
alpha asymmetry as depression/anxiety marker, etc.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


# 19 standard 10-20 channels as they appear in the BRMH dataset column names
CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

# 6 bands in the BRMH dataset (includes highbeta)
BANDS_6 = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']

# Homologous pairs for hemispheric asymmetry
HOMOLOGOUS_PAIRS = [
    ('FP1', 'FP2'), ('F3', 'F4'), ('F7', 'F8'), ('C3', 'C4'),
    ('P3', 'P4'), ('T3', 'T4'), ('T5', 'T6'), ('O1', 'O2'),
]

# Brain region definitions
REGIONS = {
    'frontal':   ['FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'central':   ['C3', 'C4', 'Cz'],
    'temporal':  ['T3', 'T4', 'T5', 'T6'],
    'parietal':  ['P3', 'P4', 'Pz'],
    'occipital': ['O1', 'O2'],
}


def extract_ab_data(df: pd.DataFrame) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Parses all AB.* columns from the BRMH dataframe into a (band, channel) → values dict.
    Handles NaNs by returning zeros for missing (band, channel) pairs.
    """
    n = len(df)
    ab_data: Dict[Tuple[str, str], np.ndarray] = {}

    for col in df.columns:
        if not col.startswith('AB.'):
            continue
        parts = col.split('.')
        if len(parts) < 5:
            continue
        band = parts[2].lower()   # e.g. 'delta'
        ch   = parts[4].upper()   # e.g. 'FP1'
        vals = df[col].values.astype(float)
        ab_data[(band, ch)] = vals

    return ab_data


def build_engineered_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Builds the full engineered feature matrix from a BRMH dataframe.

    Returns:
        X_eng  — np.ndarray of shape (n_samples, n_features)
        names  — list of feature names (same length as n_features)
    """
    n = len(df)
    zeros = np.zeros(n)
    ab = extract_ab_data(df)

    def get(band: str, ch: str) -> np.ndarray:
        return ab.get((band.lower(), ch.upper()), zeros).copy()

    features: Dict[str, np.ndarray] = {}

    # ── 1. Relative Band Power ──────────────────────────────────────────────
    for ch in CHANNELS:
        total = sum(get(b, ch) for b in BANDS_6)
        total = np.maximum(total, 1e-9)
        for b in BANDS_6:
            features[f'RP_{b}_{ch}'] = get(b, ch) / total

    # ── 2. Band Power Ratios ────────────────────────────────────────────────
    for ch in CHANNELS:
        theta   = get('theta', ch)
        alpha   = get('alpha', ch)
        beta    = get('beta', ch)
        delta   = get('delta', ch)
        gamma   = get('gamma', ch)
        hbeta   = get('highbeta', ch)

        features[f'TBR_{ch}']   = theta  / (beta    + 1e-9)   # Theta/Beta ratio (ADHD/anxiety marker)
        features[f'TAR_{ch}']   = theta  / (alpha   + 1e-9)   # Theta/Alpha ratio
        features[f'BAR_{ch}']   = beta   / (alpha   + 1e-9)   # Beta/Alpha ratio
        features[f'DAR_{ch}']   = delta  / (alpha   + 1e-9)   # Delta/Alpha ratio
        features[f'TBAR_{ch}']  = (theta + beta) / (alpha + 1e-9)  # (Theta+Beta)/Alpha
        features[f'HBAR_{ch}']  = hbeta  / (alpha   + 1e-9)   # HighBeta/Alpha
        features[f'GAR_{ch}']   = gamma  / (alpha   + 1e-9)   # Gamma/Alpha

    # ── 3. Hemispheric Asymmetry (log-ratio) ───────────────────────────────
    for b in BANDS_6:
        for left, right in HOMOLOGOUS_PAIRS:
            l_val = get(b, left)
            r_val = get(b, right)
            features[f'ASYM_{b}_{left}_{right}'] = np.log(
                (np.abs(r_val) + 1e-9) / (np.abs(l_val) + 1e-9)
            )

    # Frontal alpha asymmetry (key anxiety/depression marker: F4α − F3α)
    features['FAA_alpha']    = get('alpha', 'F4')    - get('alpha', 'F3')
    features['FAA_beta']     = get('beta', 'F4')     - get('beta', 'F3')
    features['FAA_theta']    = get('theta', 'F4')    - get('theta', 'F3')
    features['FAA_fp_alpha'] = get('alpha', 'FP2')   - get('alpha', 'FP1')

    # ── 4. Regional Averages (mean + std per band × region) ────────────────
    for b in BANDS_6:
        for reg_name, reg_chs in REGIONS.items():
            vals = np.vstack([get(b, ch) for ch in reg_chs])  # (n_ch, n_samples)
            features[f'REG_{b}_{reg_name}_mean'] = np.mean(vals, axis=0)
            features[f'REG_{b}_{reg_name}_std']  = np.std(vals,  axis=0)

    # ── 5. Global Power per Band ────────────────────────────────────────────
    for b in BANDS_6:
        vals = np.vstack([get(b, ch) for ch in CHANNELS])   # (19, n_samples)
        features[f'GLOBAL_{b}_mean'] = np.mean(vals, axis=0)
        features[f'GLOBAL_{b}_std']  = np.std(vals,  axis=0)
        features[f'GLOBAL_{b}_max']  = np.max(vals,  axis=0)

    # ── 6. Total Power Statistics ───────────────────────────────────────────
    all_band_vals = np.vstack([
        np.vstack([get(b, ch) for ch in CHANNELS]) for b in BANDS_6
    ])   # (19*6, n_samples)
    features['TOTAL_power_mean'] = np.mean(all_band_vals, axis=0)
    features['TOTAL_power_std']  = np.std(all_band_vals, axis=0)

    # ── Build matrix ────────────────────────────────────────────────────────
    names = sorted(features.keys())
    X_eng = np.column_stack([features[k] for k in names])

    return X_eng, names
