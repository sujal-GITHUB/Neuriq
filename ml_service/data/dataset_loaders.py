"""
NeuroAnxiety — Multi-Dataset Loaders
=====================================
Loads and harmonises three EEG datasets into a unified
(X: np.ndarray, y: np.ndarray, source: np.ndarray) format
where X rows are Welch PSD band-power features over the
19 standard 10-20 channels × 5 bands = 95 features.

Datasets
--------
BRMH  — Kaggle Brain-Related Mental Health (CSV, 330 samples)
         Labels: 0=Healthy, 1=Anxiety, 2=Trauma/Stress
         Features: pre-computed AB/COH columns (use directly)

DASPS — 23 subjects, 12 situations each, 14 channels, 128 Hz
         Labels derived from Hamilton Anxiety Scale scores:
           ≤17   → 0 (Healthy/Mild)
           18-24 → 1 (Anxiety)
           ≥25   → 2 (Severe/Stress)
         Processing: Welch PSD on preprocessed .mat files → 5-band powers

SEED  — 50 910 epoch samples, 5 bands × 62 channels
         Original emotion labels (0=neg, 1=neu, 2=pos) remapped:
           1 (neutral)  → 0 (Healthy proxy)
           0 (negative) → 2 (Stress proxy)   [negative affect → stress]
           2 (positive) → 0 (Healthy proxy)
         Used as AUXILIARY augmentation only (different task domain).
         Downsampled to ~500 samples to avoid overwhelming BRMH/DASPS.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import welch
from typing import Tuple, List

# ── Standard channel set ──────────────────────────────────────────────────────
STD_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz',
]
N_STD_CH  = len(STD_CHANNELS)   # 19
BANDS     = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
N_BANDS   = len(BANDS)           # 5
N_FEATURES = N_STD_CH * N_BANDS  # 95

# DASPS has 14 channels (Emotiv EPOC headset)
DASPS_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
]

# Overlap mapping DASPS → standard 10-20 positions (best approximations)
DASPS_TO_STD = {
    'AF3': 'Fp1', 'AF4': 'Fp2',
    'F7':  'F7',  'F8':  'F8',
    'F3':  'F3',  'F4':  'F4',
    'T7':  'T3',  'T8':  'T4',
    'P7':  'T5',  'P8':  'T6',
    'O1':  'O1',  'O2':  'O2',
    'FC5': 'C3',  'FC6': 'C4',
}

# SEED has 62 channels (10-20 extended). Take the 19 that overlap with std.
# The SEED dataset uses a specific 62-channel layout; we approximate by index.
# Band order in SEED: delta=0, theta=1, alpha=2, beta=3, gamma=4
SEED_BAND_ORDER = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# ── Hamilton → class mapping ──────────────────────────────────────────────────
def hamilton_to_class(score: float) -> int:
    if score <= 17: return 0    # normal / mild → Healthy
    elif score <= 24: return 1  # moderate → Anxiety
    else: return 2              # severe → Acute Stress


# ─────────────────────────────────────────────────────────────────────────────
# BRMH loader (primary dataset)
# ─────────────────────────────────────────────────────────────────────────────

def load_brmh(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads the BRMH CSV and returns raw feature matrix + feature names.
    Returns X (n, n_features), y (n,), feature_names.
    The caller is responsible for applying build_engineered_features.
    """
    TARGET = {
        'Healthy control': 0,
        'Anxiety disorder': 1,
        'Trauma and stress related disorder': 2,
    }
    df = pd.read_csv(data_path)
    df = df[df['main.disorder'].isin(TARGET.keys())].copy()
    df['label'] = df['main.disorder'].map(TARGET)
    return df, df['label'].values


# ─────────────────────────────────────────────────────────────────────────────
# DASPS loader
# ─────────────────────────────────────────────────────────────────────────────

def _welch_band_powers(signal: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Computes absolute PSD band powers via Welch's method for a single channel.
    Returns array of shape (5,) — [delta, theta, alpha, beta, gamma].
    """
    band_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    powers = []
    for low, high in band_ranges:
        mask = (freqs >= low) & (freqs < high)
        powers.append(float(np.mean(psd[mask])) if mask.any() else 0.0)
    return np.array(powers, dtype=np.float32)


def load_dasps(dasps_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads DASPS preprocessed .mat files and extracts Welch PSD features.

    Label: derived from average Hamilton score across both assessments.
    Features: 95-dim vector (19 std channels × 5 bands), DASPS channels
              mapped to closest standard 10-20 positions; unmapped channels zero-filled.

    Returns X (n_epochs, 95), y (n_epochs,)
    where n_epochs = 23 subjects × 12 situations = 276.
    """
    try:
        import mat73
    except ImportError:
        raise ImportError("mat73 is required to load DASPS .mat files: pip install mat73")

    prep_dir = os.path.join(dasps_dir, 'Preprocessed data .mat')
    raw_dir  = os.path.join(dasps_dir, 'Raw data.mat')

    if not os.path.isdir(prep_dir):
        raise FileNotFoundError(f"DASPS preprocessed dir not found: {prep_dir}")

    subjects = sorted([
        f.replace('preprocessed.mat', '')
        for f in os.listdir(prep_dir) if f.endswith('.mat')
    ])

    all_X, all_y = [], []

    for sid in subjects:
        # Load EEG data — shape (14, 1920, 12)
        eeg = mat73.loadmat(os.path.join(prep_dir, f'{sid}preprocessed.mat'))['data']
        eeg = np.array(eeg, dtype=np.float32)  # (14 ch, 1920 samples, 12 situations)

        # Load labels from raw .mat
        raw    = mat73.loadmat(os.path.join(raw_dir, f'{sid}.mat'))
        ham    = np.array(raw['hamilton'], dtype=float)   # [ham1, ham2]
        avg_ham = float(np.mean(ham))
        label  = hamilton_to_class(avg_ham)

        n_situations = eeg.shape[2]   # 12
        n_channels   = eeg.shape[0]   # 14

        for sit_idx in range(n_situations):
            epoch = eeg[:, :, sit_idx]   # (14, 1920)

            # Build 95-dim feature vector: zero-fill unmapped channels
            feat = np.zeros((N_STD_CH, N_BANDS), dtype=np.float32)

            for ch_idx, ch_name in enumerate(DASPS_CHANNELS):
                std_ch = DASPS_TO_STD.get(ch_name)
                if std_ch is None:
                    continue
                std_idx = STD_CHANNELS.index(std_ch)
                signal = epoch[ch_idx]
                feat[std_idx] = _welch_band_powers(signal, fs=128)

            all_X.append(feat.flatten())   # (95,)
            all_y.append(label)

    X = np.array(all_X, dtype=np.float32)   # (276, 95)
    y = np.array(all_y, dtype=np.int64)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# SEED loader
# ─────────────────────────────────────────────────────────────────────────────

def load_seed(seed_dir: str, max_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the SEED .npz files and maps emotion labels to anxiety proxies.

    SEED feature shape: (50910, 5_bands, 62_channels)
    We take the first 19 channels as a proxy for standard 10-20 positions
    (SEED uses an extended 10-20 layout where channels 0-18 correspond
    roughly to standard frontal/central/parietal/occipital positions).

    Label remapping (emotion → anxiety proxy):
        0 (negative affect) → 2 (Acute Stress)
        1 (neutral)         → 0 (Healthy)
        2 (positive affect) → 0 (Healthy)

    max_samples: cap to avoid overwhelming the smaller BRMH/DASPS sets.
                 Samples are drawn in a class-balanced manner.
    """
    X_raw = np.load(os.path.join(seed_dir, 'DatasetCaricatoNoImage.npz'))['arr_0']
    L_raw = np.load(os.path.join(seed_dir, 'LabelsNoImage.npz'))['arr_0']

    # Remap labels
    remap = {0: 2, 1: 0, 2: 0}
    y = np.array([remap[int(l)] for l in L_raw], dtype=np.int64)

    # Take first 19 channels across all 5 bands → (n, 5, 19) → (n, 95)
    X = X_raw[:, :, :N_STD_CH]          # (n, 5, 19)
    X = X.reshape(X.shape[0], -1)       # (n, 95)  — bands vary faster than channels

    # Reorder to match our channel-major layout: (ch, band) → flatten
    # SEED stores (band, channel), we want (channel, band)
    X_full = X_raw[:, :, :N_STD_CH]            # (n, 5_bands, 19_ch)
    X_full = X_full.transpose(0, 2, 1)         # (n, 19_ch, 5_bands)
    X = X_full.reshape(X_full.shape[0], -1)    # (n, 95)

    # Balanced downsample
    classes = np.unique(y)
    per_class = max_samples // len(classes)
    idx_keep = []
    rng = np.random.default_rng(42)
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        chosen  = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
        idx_keep.append(chosen)
    idx_keep = np.concatenate(idx_keep)
    rng.shuffle(idx_keep)

    X = X[idx_keep].astype(np.float32)
    y = y[idx_keep]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Combined loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_datasets(
    brmh_path:  str,
    dasps_dir:  str,
    seed_dir:   str,
    use_dasps:  bool = True,
    use_seed:   bool = True,
    seed_max_samples: int = 600,
    verbose:    bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Loads and combines BRMH + (optionally) DASPS + SEED.

    Returns:
        df_brmh    — raw BRMH DataFrame (for build_engineered_features)
        X_extra    — (n_extra, 95) array of DASPS + SEED features (band powers only)
        y_extra    — (n_extra,) labels for extra samples
        sources    — list of source strings per extra sample

    The caller should merge X_extra with the BRMH feature matrix after
    feature engineering (DASPS/SEED rows get zeros for COH/engineered columns).
    """
    # ── BRMH ──────────────────────────────────────────────────────────────
    if verbose: print("  [BRMH] Loading primary dataset…")
    TARGET = {
        'Healthy control': 0,
        'Anxiety disorder': 1,
        'Trauma and stress related disorder': 2,
    }
    df_brmh = pd.read_csv(brmh_path)
    df_brmh = df_brmh[df_brmh['main.disorder'].isin(TARGET.keys())].copy()
    df_brmh['label'] = df_brmh['main.disorder'].map(TARGET)
    if verbose:
        counts = dict(zip(*np.unique(df_brmh['label'].values, return_counts=True)))
        print(f"  [BRMH] {len(df_brmh)} samples | classes {counts}")

    extra_X, extra_y, extra_sources = [], [], []

    # ── DASPS ──────────────────────────────────────────────────────────────
    if use_dasps and os.path.isdir(dasps_dir):
        try:
            if verbose: print("  [DASPS] Loading 23 subjects × 12 situations…")
            X_d, y_d = load_dasps(dasps_dir)
            counts = dict(zip(*np.unique(y_d, return_counts=True)))
            if verbose: print(f"  [DASPS] {len(y_d)} epochs | classes {counts}")
            extra_X.append(X_d)
            extra_y.append(y_d)
            extra_sources.extend(['DASPS'] * len(y_d))
        except Exception as e:
            warnings.warn(f"DASPS loading failed, skipping: {e}")
    elif use_dasps:
        warnings.warn(f"DASPS directory not found at {dasps_dir}, skipping.")

    # ── SEED ───────────────────────────────────────────────────────────────
    if use_seed and os.path.isdir(seed_dir):
        try:
            if verbose: print(f"  [SEED] Loading (max {seed_max_samples} samples)…")
            X_s, y_s = load_seed(seed_dir, max_samples=seed_max_samples)
            counts = dict(zip(*np.unique(y_s, return_counts=True)))
            if verbose: print(f"  [SEED] {len(y_s)} samples | classes {counts}")
            extra_X.append(X_s)
            extra_y.append(y_s)
            extra_sources.extend(['SEED'] * len(y_s))
        except Exception as e:
            warnings.warn(f"SEED loading failed, skipping: {e}")
    elif use_seed:
        warnings.warn(f"SEED directory not found at {seed_dir}, skipping.")

    if extra_X:
        X_extra = np.vstack(extra_X).astype(np.float32)
        y_extra = np.concatenate(extra_y).astype(np.int64)
    else:
        X_extra  = np.empty((0, N_FEATURES), dtype=np.float32)
        y_extra  = np.empty((0,), dtype=np.int64)

    return df_brmh, X_extra, y_extra, extra_sources
