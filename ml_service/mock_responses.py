"""
Mock API Responses for NeuroAnxiety ML Service
===============================================
Realistic mock data so the frontend can be developed independently.
"""

import random
import time
import uuid

def mock_prediction_response(model_used: str = "brain2vec") -> dict:
    """Return a realistic mock prediction result."""
    anxiety_levels = ["Low", "Moderate", "High"]
    anxiety_idx = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
    anxiety_level = anxiety_levels[anxiety_idx]
    
    # Generate probabilities that sum to 1
    probs = [random.random() for _ in range(3)]
    probs[anxiety_idx] += 2.0  # Bias toward selected class
    total = sum(probs)
    probs = [p / total for p in probs]
    
    return {
        "anxiety_level": anxiety_level,
        "confidence": max(probs),
        "probabilities": {
            "Low": round(probs[0], 4),
            "Moderate": round(probs[1], 4),
            "High": round(probs[2], 4)
        },
        "top_features": [
            {"name": "Beta_Power_F4", "value": round(random.uniform(5, 25), 3), "importance": 0.142},
            {"name": "Alpha_Power_F3", "value": round(random.uniform(8, 30), 3), "importance": 0.128},
            {"name": "Frontal_Alpha_Asymmetry", "value": round(random.uniform(-0.5, 0.5), 4), "importance": 0.115},
            {"name": "Alpha_Beta_Ratio_Cz", "value": round(random.uniform(0.3, 2.5), 3), "importance": 0.098},
            {"name": "Theta_Power_Fz", "value": round(random.uniform(3, 15), 3), "importance": 0.091},
            {"name": "Sample_Entropy_C3", "value": round(random.uniform(0.5, 2.0), 4), "importance": 0.084},
            {"name": "Hjorth_Complexity_P3", "value": round(random.uniform(1.0, 3.0), 4), "importance": 0.078},
            {"name": "Gamma_Power_T7", "value": round(random.uniform(1, 10), 3), "importance": 0.072},
            {"name": "Delta_Power_O1", "value": round(random.uniform(10, 50), 3), "importance": 0.065},
            {"name": "Spectral_Entropy_Fp1", "value": round(random.uniform(0.6, 0.95), 4), "importance": 0.058},
            {"name": "RMS_C4", "value": round(random.uniform(5, 30), 3), "importance": 0.052},
            {"name": "Permutation_Entropy_P4", "value": round(random.uniform(0.8, 1.0), 4), "importance": 0.047},
            {"name": "Kurtosis_F7", "value": round(random.uniform(-1, 5), 3), "importance": 0.043},
            {"name": "Theta_Alpha_Ratio_Fz", "value": round(random.uniform(0.5, 3.0), 3), "importance": 0.039},
            {"name": "Zero_Crossing_Rate_T8", "value": round(random.uniform(30, 80), 1), "importance": 0.035},
            {"name": "DWT_Energy_D3_Cz", "value": round(random.uniform(100, 800), 2), "importance": 0.031},
            {"name": "Higuchi_FD_AF3", "value": round(random.uniform(1.2, 1.8), 4), "importance": 0.028},
            {"name": "Peak_to_Peak_CP5", "value": round(random.uniform(20, 100), 2), "importance": 0.025},
            {"name": "Variance_FC1", "value": round(random.uniform(10, 80), 2), "importance": 0.022},
            {"name": "DFA_Exponent_O2", "value": round(random.uniform(0.5, 1.5), 4), "importance": 0.019}
        ],
        "band_powers": {
            "delta": round(random.uniform(15, 40), 2),
            "theta": round(random.uniform(8, 20), 2),
            "alpha": round(random.uniform(10, 35), 2),
            "beta": round(random.uniform(8, 25), 2),
            "gamma": round(random.uniform(2, 12), 2)
        },
        "alpha_beta_ratio": round(random.uniform(0.4, 2.5), 3),
        "frontal_asymmetry": round(random.uniform(-0.5, 0.5), 4),
        "model_used": model_used,
        "inference_time_ms": round(random.uniform(15, 150), 2)
    }


def mock_training_response() -> dict:
    """Return mock training initiation response."""
    return {
        "status": "started",
        "job_id": str(uuid.uuid4())
    }


def mock_training_status(progress: float = None) -> dict:
    """Return mock training status."""
    if progress is None:
        progress = round(random.uniform(0, 100), 1)
    
    epoch = int(progress)
    return {
        "status": "running" if progress < 100 else "completed",
        "progress": progress,
        "current_epoch": epoch,
        "total_epochs": 100,
        "current_metrics": {
            "train_loss": round(1.2 - (progress / 100) * 0.9 + random.uniform(-0.05, 0.05), 4),
            "val_loss": round(1.3 - (progress / 100) * 0.8 + random.uniform(-0.08, 0.08), 4),
            "train_accuracy": round(0.35 + (progress / 100) * 0.55 + random.uniform(-0.03, 0.03), 4),
            "val_accuracy": round(0.30 + (progress / 100) * 0.50 + random.uniform(-0.05, 0.05), 4),
            "learning_rate": 1e-4 * (1 - progress / 200)
        },
        "log": [
            f"Epoch {epoch}/100 - loss: {round(1.2 - (progress/100)*0.9, 4)} - acc: {round(0.35 + (progress/100)*0.55, 4)}"
        ]
    }


def mock_metrics_response(model_name: str = "brain2vec", dataset: str = "DEAP") -> dict:
    """Return comprehensive mock metrics for a model/dataset combo."""
    base_acc = {
        "brain2vec": 0.89, "cnn_lstm": 0.86, "eegnet": 0.83,
        "ensemble": 0.91, "svm": 0.78, "random_forest": 0.82,
        "xgboost": 0.85, "knn": 0.74, "lda": 0.72
    }.get(model_name, 0.80)
    
    dataset_modifier = {"DASPS": 0.0, "DEAP": -0.02, "MAHNOB": -0.04}.get(dataset, 0.0)
    acc = min(base_acc + dataset_modifier + random.uniform(-0.02, 0.02), 0.99)
    
    def gen_per_class(base):
        return {
            "Low": round(base + random.uniform(-0.05, 0.05), 4),
            "Moderate": round(base - 0.03 + random.uniform(-0.05, 0.05), 4),
            "High": round(base + 0.02 + random.uniform(-0.05, 0.05), 4)
        }
    
    n_folds = 5
    fold_accs = [round(acc + random.uniform(-0.04, 0.04), 4) for _ in range(n_folds)]
    
    return {
        "model_name": model_name,
        "dataset": dataset,
        "per_class": {
            "precision": gen_per_class(acc),
            "recall": gen_per_class(acc - 0.02),
            "f1_score": gen_per_class(acc - 0.01),
            "support": {"Low": 420, "Moderate": 380, "High": 400},
            "specificity": gen_per_class(acc + 0.05)
        },
        "overall": {
            "accuracy": round(acc, 4),
            "macro_f1": round(acc - 0.01, 4),
            "weighted_f1": round(acc, 4),
            "cohens_kappa": round(acc - 0.08, 4),
            "mcc": round(acc - 0.06, 4),
            "balanced_accuracy": round(acc - 0.01, 4),
            "roc_auc": {
                "Low": round(acc + 0.05, 4),
                "Moderate": round(acc + 0.03, 4),
                "High": round(acc + 0.06, 4),
                "macro": round(acc + 0.045, 4)
            },
            "average_precision": gen_per_class(acc + 0.02)
        },
        "confusion_matrix": {
            "raw": [
                [int(400 * acc), int(400 * (1-acc) * 0.6), int(400 * (1-acc) * 0.4)],
                [int(380 * (1-acc) * 0.5), int(380 * acc), int(380 * (1-acc) * 0.5)],
                [int(400 * (1-acc) * 0.3), int(400 * (1-acc) * 0.7), int(400 * acc)]
            ],
            "normalized": [
                [round(acc, 3), round((1-acc)*0.6, 3), round((1-acc)*0.4, 3)],
                [round((1-acc)*0.5, 3), round(acc, 3), round((1-acc)*0.5, 3)],
                [round((1-acc)*0.3, 3), round((1-acc)*0.7, 3), round(acc, 3)]
            ],
            "labels": ["Low", "Moderate", "High"]
        },
        "roc_curves": {
            "Low": {
                "fpr": [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
                "tpr": [0, 0.4, 0.65, 0.78, 0.85, 0.9, 0.94, 0.97, 0.99, 1.0]
            },
            "Moderate": {
                "fpr": [0, 0.03, 0.08, 0.12, 0.18, 0.25, 0.35, 0.55, 0.75, 1.0],
                "tpr": [0, 0.35, 0.58, 0.72, 0.82, 0.87, 0.92, 0.96, 0.98, 1.0]
            },
            "High": {
                "fpr": [0, 0.01, 0.04, 0.08, 0.13, 0.18, 0.28, 0.48, 0.68, 1.0],
                "tpr": [0, 0.45, 0.7, 0.82, 0.88, 0.92, 0.95, 0.98, 0.99, 1.0]
            }
        },
        "cross_validation": {
            "fold_accuracies": fold_accs,
            "mean_accuracy": round(sum(fold_accs) / n_folds, 4),
            "std_accuracy": round(max(0.01, (max(fold_accs) - min(fold_accs)) / 4), 4),
            "fold_f1s": [round(a - 0.01, 4) for a in fold_accs],
            "training_times": [round(random.uniform(45, 120), 1) for _ in range(n_folds)],
            "inference_time_ms": round(random.uniform(5, 50), 2)
        },
        "statistical_tests": {
            "mcnemar": {"statistic": round(random.uniform(0.5, 5), 3), "p_value": round(random.uniform(0.01, 0.5), 4)},
            "friedman": {"statistic": round(random.uniform(5, 25), 3), "p_value": round(random.uniform(0.001, 0.1), 4)}
        },
        "shap": {
            "top_features": [
                {"name": "Beta_Power_F4", "channel": "F4", "band": "Beta", "mean_shap": 0.142, "direction": "positive"},
                {"name": "Alpha_Power_F3", "channel": "F3", "band": "Alpha", "mean_shap": 0.128, "direction": "negative"},
                {"name": "Frontal_Asymmetry", "channel": "F3-F4", "band": "Alpha", "mean_shap": 0.115, "direction": "positive"},
                {"name": "Alpha_Beta_Ratio_Cz", "channel": "Cz", "band": "Alpha/Beta", "mean_shap": 0.098, "direction": "negative"},
                {"name": "Theta_Power_Fz", "channel": "Fz", "band": "Theta", "mean_shap": 0.091, "direction": "positive"},
                {"name": "Sample_Entropy_C3", "channel": "C3", "band": "N/A", "mean_shap": 0.084, "direction": "positive"},
                {"name": "Hjorth_Complexity_P3", "channel": "P3", "band": "N/A", "mean_shap": 0.078, "direction": "negative"},
                {"name": "Gamma_Power_T7", "channel": "T7", "band": "Gamma", "mean_shap": 0.072, "direction": "positive"},
                {"name": "Delta_Power_O1", "channel": "O1", "band": "Delta", "mean_shap": 0.065, "direction": "negative"},
                {"name": "Spectral_Entropy_Fp1", "channel": "Fp1", "band": "N/A", "mean_shap": 0.058, "direction": "positive"},
                {"name": "RMS_C4", "channel": "C4", "band": "N/A", "mean_shap": 0.052, "direction": "positive"},
                {"name": "Permutation_Entropy_P4", "channel": "P4", "band": "N/A", "mean_shap": 0.047, "direction": "negative"},
                {"name": "Kurtosis_F7", "channel": "F7", "band": "N/A", "mean_shap": 0.043, "direction": "positive"},
                {"name": "Theta_Alpha_Ratio_Fz", "channel": "Fz", "band": "Theta/Alpha", "mean_shap": 0.039, "direction": "positive"},
                {"name": "Zero_Crossing_Rate_T8", "channel": "T8", "band": "N/A", "mean_shap": 0.035, "direction": "negative"},
                {"name": "DWT_Energy_D3_Cz", "channel": "Cz", "band": "DWT-D3", "mean_shap": 0.031, "direction": "positive"},
                {"name": "Higuchi_FD_AF3", "channel": "AF3", "band": "N/A", "mean_shap": 0.028, "direction": "positive"},
                {"name": "Peak_to_Peak_CP5", "channel": "CP5", "band": "N/A", "mean_shap": 0.025, "direction": "negative"},
                {"name": "Variance_FC1", "channel": "FC1", "band": "N/A", "mean_shap": 0.022, "direction": "positive"},
                {"name": "DFA_Exponent_O2", "channel": "O2", "band": "N/A", "mean_shap": 0.019, "direction": "positive"}
            ]
        }
    }


def mock_models_list() -> dict:
    """Return list of available model checkpoints."""
    return {
        "models": [
            {
                "name": "brain2vec",
                "display_name": "Brain2Vec (CNN+LSTM+Attention)",
                "type": "deep",
                "framework": "PyTorch",
                "parameters": 2_450_000,
                "description": "CNN + LSTM + Self-Attention hybrid from arXiv:2506.11179",
                "trained_on": ["DASPS"],
                "best_accuracy": 0.891,
                "checkpoint_path": "models/checkpoints/brain2vec_dasps_best.pt"
            },
            {
                "name": "cnn_lstm",
                "display_name": "CNN-BiLSTM with Attention",
                "type": "deep",
                "framework": "PyTorch",
                "parameters": 3_200_000,
                "description": "Convolutional + Bidirectional LSTM with attention mechanism",
                "trained_on": ["DASPS"],
                "best_accuracy": 0.864,
                "checkpoint_path": "models/checkpoints/cnn_lstm_dasps_best.pt"
            },
            {
                "name": "eegnet",
                "display_name": "EEGNet",
                "type": "deep",
                "framework": "PyTorch",
                "parameters": 890_000,
                "description": "Compact CNN for EEG from Lawhern et al., 2018",
                "trained_on": ["DASPS"],
                "best_accuracy": 0.832,
                "checkpoint_path": "models/checkpoints/eegnet_dasps_best.pt"
            },
            {
                "name": "boosting_ensemble",
                "display_name": "Boosting Ensemble (XGB+LGBM+CatBoost)",
                "type": "classical",
                "framework": "Optuna + Boosters",
                "parameters": None,
                "description": "XGBoost, LightGBM, and CatBoost soft voting ensemble tuned with Optuna",
                "trained_on": ["DASPS"],
                "best_accuracy": 0.9733,
                "checkpoint_path": "models/checkpoints/boosting_ensemble_best.joblib"
            },
            {
                "name": "ensemble",
                "display_name": "Classical Ensemble (Meta-Classifier)",
                "type": "classical",
                "framework": "scikit-learn",
                "parameters": None,
                "description": "SVM + RF + XGBoost + KNN + LDA with Logistic Regression meta-classifier",
                "trained_on": ["DASPS"],
                "best_accuracy": 0.912,
                "checkpoint_path": "models/checkpoints/ensemble_dasps_best.joblib"
            }
        ]
    }


def mock_datasets_info() -> dict:
    """Return info about available datasets."""
    return {
        "datasets": [
            {
                "name": "DASPS",
                "full_name": "Dataset for Anxiety using Psychological Stimuli",
                "loaded": True,
                "num_subjects": 23,
                "num_trials": 6,
                "num_eeg_channels": 14,
                "num_peripheral_channels": 0,
                "sampling_rate": 128,
                "trial_duration_sec": 60,
                "labels": ["loss", "family issues", "financial issues", "deadline", "witnessing deadly accident", "mistreating"],
                "label_scale": "Anxiety Elicitation %",
                "modalities": ["EEG"],
                "format": ".csv / .edf",
                "download_url": "Private / Not publicly available via URL",
                "reference": "Eudoxuspress"
            },
            {
                "name": "DREAMER",
                "full_name": "DREAMER: A Database for Emotion Recognition",
                "loaded": False,
                "num_subjects": 23,
                "num_trials": 18,
                "num_eeg_channels": 14,
                "num_peripheral_channels": 2,
                "sampling_rate": 128,
                "trial_duration_sec": 60,
                "labels": ["valence", "arousal", "dominance"],
                "label_scale": "1-5",
                "modalities": ["EEG", "ECG"],
                "format": ".mat (MATLAB)",
                "download_url": "https://zenodo.org/record/546113",
                "reference": "Katsigiannis & Ramzan, 2018"
            },
            {
                "name": "MAHNOB-HCI",
                "full_name": "MAHNOB-HCI: Multimodal Database for Affect Recognition",
                "loaded": False,
                "num_subjects": 30,
                "num_trials": 20,
                "num_eeg_channels": 32,
                "num_peripheral_channels": 6,
                "sampling_rate": 256,
                "trial_duration_sec": 60,
                "labels": ["arousal", "valence"],
                "label_scale": "1-9",
                "modalities": ["EEG", "GSR", "ECG", "Temperature", "Respiration", "Video", "Gaze"],
                "format": ".bdf + .xml",
                "download_url": "https://mahnob-db.eu/hci-tagging/",
                "reference": "Soleymani et al., 2012"
            }
        ]
    }


def mock_upload_response(filename: str) -> dict:
    """Return mock parsed EEG data from file upload."""
    import numpy as np
    
    channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    n_channels = len(channels)
    n_samples = 128 * 30  # 30 seconds at 128 Hz
    
    # Generate synthetic EEG-like signals
    t = np.linspace(0, 30, n_samples)
    signals = []
    for i in range(n_channels):
        alpha = 15 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
        beta = 8 * np.sin(2 * np.pi * 22 * t + np.random.uniform(0, 2*np.pi))
        noise = np.random.normal(0, 5, n_samples)
        signals.append((alpha + beta + noise).tolist())
    
    return {
        "filename": filename,
        "channels": channels,
        "sampling_rate": 128,
        "duration_sec": 30.0,
        "num_samples": n_samples,
        "signals": signals
    }
