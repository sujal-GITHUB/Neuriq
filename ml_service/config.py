"""
NeuroAnxiety ML Service Configuration
=====================================
Central configuration for paths, hyperparameters, and constants.
"""

import os
from pathlib import Path

# ── Base Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "datasets"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset Paths ───────────────────────────────────────────────────
DEAP_DIR = DATA_DIR / "DEAP" / "data_preprocessed_python"
DREAMER_PATH = DATA_DIR / "DREAMER" / "DREAMER.mat"
MAHNOB_DIR = DATA_DIR / "MAHNOB-HCI"

# ── EEG Channel Configurations ─────────────────────────────────────
DEAP_EEG_CHANNELS = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
    'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz'
]

DREAMER_EEG_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

MAHNOB_EEG_CHANNELS = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
    'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz'
]

# Frontal channels for asymmetry
FRONTAL_PAIRS = {
    'F3': 'F4',
    'Fp1': 'Fp2',
    'AF3': 'AF4',
    'FC1': 'FC2',
    'FC5': 'FC6'
}

# ── Preprocessing Parameters ───────────────────────────────────────
SAMPLING_RATE = 128          # Target sampling rate (Hz)
BANDPASS_LOW = 0.5           # Bandpass low cutoff (Hz)
BANDPASS_HIGH = 45.0         # Bandpass high cutoff (Hz)
NOTCH_FREQ = 50.0            # Power-line noise frequency (Hz)
FILTER_ORDER = 5             # Butterworth filter order
ICA_N_COMPONENTS = 20        # Number of ICA components
EPOCH_DURATION = 4           # Epoch length in seconds
EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_DURATION  # 512 samples
BASELINE_DURATION = 3        # Pre-stimulus baseline (seconds)

# ── EEG Frequency Bands ────────────────────────────────────────────
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# ── Anxiety Labels ──────────────────────────────────────────────────
ANXIETY_LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}
NUM_CLASSES = 3

# ── Model Hyperparameters ──────────────────────────────────────────
# Classical models
SVM_C = 10
SVM_GAMMA = 'scale'
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 10
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
KNN_N_NEIGHBORS = 7
KNN_METRIC = 'euclidean'

# Deep learning
BATCH_SIZE = 64
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.1
CV_FOLDS = 5

# Brain2Vec specific
B2V_CONV_FILTERS = [32, 64, 128]
B2V_LSTM_HIDDEN = 256
B2V_DROPOUT = 0.4

# CNN-LSTM specific
CL_CONV1_FILTERS = 64
CL_CONV2_FILTERS = 128
CL_LSTM_HIDDEN = 256
CL_LSTM_LAYERS = 2
CL_LSTM_DROPOUT = 0.3

# EEGNet specific
EEGNET_F1 = 8
EEGNET_F2 = 16
EEGNET_D = 2
EEGNET_DROPOUT = 0.5

# Feature selection
FEATURE_SELECTION_K = 200     # SelectKBest k
VARIANCE_THRESHOLD = 0.01

# ── API Configuration ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]
