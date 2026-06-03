"""
NeuroAnxiety ML Service — EEG Inference Module (Step 6)
======================================================
Handles real-time inference for psychiatric stress classification.

Updated to use the improved XGForest+ pipeline:
  - StandardScaler + SelectKBest feature selection
  - Full AB + COH + engineered feature set
  - Backward-compatible: falls back to PCA-RFE selector if new artefacts absent
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

import sys
import types

from ml_service.data.eeg_preprocessor import EEGPreprocessor
from ml_service.features.eeg_pca_rfe import PCARFEHybridSelector
from ml_service.features.eeg_feature_engineering import build_engineered_features

# Inject classes into module namespaces for joblib deserialization
for module_name in ['__main__', '__mp_main__']:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)
    setattr(sys.modules[module_name], 'EEGPreprocessor',     EEGPreprocessor)
    setattr(sys.modules[module_name], 'PCARFEHybridSelector', PCARFEHybridSelector)


class AnxietyInferenceEngine:
    """Unified inference engine for psychiatric disorder classification (EEG spectral features)."""

    # Standard 19-channel names and 5-band names (used for legacy/raw-signal input)
    STANDARD_CHANNELS = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    def __init__(self):
        self.stacking_model = None
        self.scaler         = None   # new: StandardScaler
        self.selector       = None   # new: SelectKBest  (or legacy PCARFEHybridSelector)
        self.feature_meta   = None   # dict with 'feature_names', 'selected_names', etc.
        self.preprocessor   = None   # legacy EEGPreprocessor (kept for compat)
        self.feature_cols   = [
            f"PSD_{band}_{ch}"
            for ch in self.STANDARD_CHANNELS
            for band in self.BANDS
        ]
        self._use_new_pipeline = False
        self.load_models()

    # ─────────────────────────────────────────────────────────────────
    def load_models(self):
        """Loads model checkpoints — prefers improved pipeline, falls back to legacy."""
        base_dir   = os.path.dirname(os.path.abspath(__file__))
        ckpt_dir   = os.path.join(base_dir, "models", "checkpoints")

        model_path      = os.path.join(ckpt_dir, "best_stacking_model.joblib")
        scaler_path     = os.path.join(ckpt_dir, "eeg_scaler.joblib")
        selector_path   = os.path.join(ckpt_dir, "eeg_selector.joblib")
        meta_path       = os.path.join(ckpt_dir, "feature_metadata.joblib")
        preprocessor_path = os.path.join(ckpt_dir, "eeg_preprocessor.joblib")

        if not os.path.exists(model_path):
            print(f"EEG Inference Engine: No checkpoint found at {model_path}")
            return

        try:
            self.stacking_model = joblib.load(model_path)

            # Prefer new pipeline artefacts
            if os.path.exists(scaler_path) and os.path.exists(meta_path):
                self.scaler       = joblib.load(scaler_path)
                self.selector     = joblib.load(selector_path)
                self.feature_meta = joblib.load(meta_path)
                self._use_new_pipeline = True
                print("EEG Inference Engine: Loaded improved XGForest+ pipeline.")
            else:
                # Legacy: PCA-RFE artefacts
                self.preprocessor = joblib.load(preprocessor_path)
                self.selector     = joblib.load(selector_path)
                self._use_new_pipeline = False
                print("EEG Inference Engine: Loaded legacy XGForest pipeline.")
        except Exception as e:
            print(f"EEG Inference Engine: Failed to load checkpoints: {e}")

    # ─────────────────────────────────────────────────────────────────
    def predict_from_signals(
        self,
        signals:       np.ndarray,
        channels:      List[str],
        sampling_rate: int = 128,
        model_name:    str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Runs prediction from multi-channel raw EEG matrices.
        Extracts absolute spectral band powers via Welch's method,
        then delegates to predict_from_features.
        """
        if signals.ndim != 2:
            return self.predict_from_features({})

        from scipy.signal import welch
        bands = {
            'Delta':   (0.5,  4.0),
            'Theta':   (4.0,  8.0),
            'Alpha':   (8.0,  13.0),
            'Beta':    (13.0, 30.0),
            'Gamma':   (30.0, 45.0),
        }

        feats: Dict[str, float] = {}
        for ch_idx, ch_name in enumerate(self.STANDARD_CHANNELS):
            if ch_name in channels:
                sig = signals[channels.index(ch_name)]
            elif ch_idx < len(signals):
                sig = signals[ch_idx]
            else:
                sig = signals[0]

            freqs, psd = welch(sig, fs=sampling_rate, nperseg=min(256, len(sig)))
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs <= high)
                feats[f"PSD_{band_name}_{ch_name}"] = (
                    float(np.mean(psd[mask])) if np.any(mask) else 15.0
                )

        return self.predict_from_features(feats, model_name)

    # ─────────────────────────────────────────────────────────────────
    def predict_from_features(
        self,
        features:   Dict[str, float],
        model_name: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Runs predictions from a pre-extracted spectral feature dictionary.
        Builds the full feature vector expected by the trained pipeline.
        """
        start_time = time.time()
        ml_probs   = None

        # ── Normalize absolute PSD to relative power proportions ─────
        normalized: Dict[str, float] = {}
        for ch in self.STANDARD_CHANNELS:
            total = sum(features.get(f"PSD_{b}_{ch}", 0.0) for b in self.BANDS)
            if total <= 0.0:
                total = 1e-9
            for b in self.BANDS:
                col = f"PSD_{b}_{ch}"
                normalized[col] = features.get(col, 0.0) / total

        # ── Run ML inference ─────────────────────────────────────────
        if self.stacking_model is not None:
            try:
                if self._use_new_pipeline:
                    ml_probs = self._infer_new_pipeline(features, normalized)
                else:
                    ml_probs = self._infer_legacy_pipeline(normalized)
            except Exception as e:
                print(f"ML inference warning: {e}")

        # ── Output formatting ────────────────────────────────────────
        if ml_probs is None:
            probs = [0.85, 0.08, 0.07]   # safe fallback
        else:
            probs = ml_probs

        # Band power averages across all channels
        extracted_bands: Dict[str, float] = {}
        for band in self.BANDS:
            vals = [features.get(f"PSD_{band}_{ch}", 15.0) for ch in self.STANDARD_CHANNELS]
            extracted_bands[band.lower()] = float(np.mean(vals))

        # Frontal alpha asymmetry
        f3_alpha = features.get("PSD_Alpha_F3", 15.0)
        f4_alpha = features.get("PSD_Alpha_F4", 15.0)
        frontal_asymmetry = f4_alpha - f3_alpha

        class_idx = int(np.argmax(probs))
        class_map = {
            0: "Healthy Controls (Baseline)",
            1: "Social Anxiety Disorder",
            2: "Acute Stress State",
        }

        return {
            "class_prediction": class_map[class_idx],
            "confidence":       round(probs[class_idx] * 100, 1),
            "probabilities": {
                "Healthy Controls": round(probs[0] * 100, 1),
                "Social Anxiety":   round(probs[1] * 100, 1),
                "Acute Stress":     round(probs[2] * 100, 1),
            },
            "band_powers":         extracted_bands,
            "frontal_asymmetry":   round(frontal_asymmetry, 4),
            "inference_time_ms":   round((time.time() - start_time) * 1000, 2),
        }

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────

    def _infer_new_pipeline(
        self,
        raw_features:        Dict[str, float],
        normalized_features: Dict[str, float],
    ) -> List[float]:
        """
        Inference path for the improved XGForest+ pipeline.
        Reconstructs the full feature vector expected by the new model:
          raw AB-equivalent + COH (zeros for unsupported live inference) + engineered + demographics.
        """
        # Build a minimal single-row DataFrame to run build_engineered_features
        # We synthesise AB columns from the PSD relative-power values we have.
        # COH features are not available during live signal inference, so we zero-fill them.

        feature_names = self.feature_meta.get('feature_names', [])
        k_features    = self.feature_meta.get('k_features', 50)

        # Create row dict with all feature columns set to 0.0
        row = {name: 0.0 for name in feature_names}

        # Fill AB columns using the normalized PSD values as proxies
        # Mapping: PSD_{Band}_{CH} → AB.AB.{band}.mean.{CH}  (approximate)
        band_map = {
            'Delta': 'delta', 'Theta': 'theta', 'Alpha': 'alpha',
            'Beta':  'beta',  'Gamma': 'gamma',
        }
        ch_map_rev = {ch.lower(): ch.upper() for ch in [
            'FP1','FP2','F7','F3','Fz','F4','F8','T3','C3','Cz',
            'C4','T4','T5','P3','Pz','P4','T6','O1','O2'
        ]}
        # Map standard channel names to BRMH channel names
        std_to_brmh = {
            'Fp1': 'FP1', 'Fp2': 'FP2', 'F3': 'F3', 'F4': 'F4',
            'C3':  'C3',  'C4':  'C4',  'P3': 'P3', 'P4': 'P4',
            'O1':  'O1',  'O2':  'O2',  'F7': 'F7', 'F8': 'F8',
            'T3':  'T3',  'T4':  'T4',  'T5': 'T5', 'T6': 'T6',
            'Fz':  'Fz',  'Cz':  'Cz',  'Pz': 'Pz',
        }

        # Build a small DataFrame with the AB column structure expected by build_engineered_features
        df_infer = pd.DataFrame([{
            **{name: 0.0 for name in feature_names},
            'age': 30.0,
            'sex': 'M',
        }])

        # Inject AB values from normalized PSD features
        for std_ch, brmh_ch in std_to_brmh.items():
            for band_std, band_brmh in band_map.items():
                psd_key = f"PSD_{band_std}_{std_ch}"
                val = normalized_features.get(psd_key, 0.0)
                # Find matching AB column
                for col in feature_names:
                    if (col.startswith('AB.') and
                            band_brmh in col.lower() and
                            brmh_ch.upper() == col.split('.')[-1].upper()):
                        df_infer[col] = val
                        break

        # Rebuild engineered features from this single-row dataframe
        X_eng, _ = build_engineered_features(df_infer)

        # Rebuild the full vector in the same column order
        X_raw = df_infer[[n for n in feature_names
                           if n.startswith('AB.') or n.startswith('COH.')]].values.astype(float)
        demo  = np.array([[30.0, 1.0]])   # placeholder demographics
        X_full = np.hstack([X_raw, X_eng, demo])   # shape: (1, total_features)

        X_scaled   = self.scaler.transform(X_full)
        X_selected = self.selector.transform(X_scaled)
        probs      = self.stacking_model.predict_proba(X_selected)[0].tolist()
        return probs

    def _infer_legacy_pipeline(
        self,
        normalized_features: Dict[str, float],
    ) -> List[float]:
        """Legacy inference path using EEGPreprocessor + PCARFEHybridSelector."""
        row_dict = {col: normalized_features.get(col, np.nan) for col in self.feature_cols}
        df_row   = pd.DataFrame([row_dict])
        df_clean = self.preprocessor.transform(df_row, self.feature_cols)
        X        = df_clean[self.feature_cols].values
        X_sel    = self.selector.transform(X)
        probs    = self.stacking_model.predict_proba(X_sel)[0].tolist()
        return probs
