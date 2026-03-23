"""
NeuroAnxiety — Dataset Loaders
==============================
Loaders for DEAP, DREAMER, and MAHNOB-HCI datasets.
Each loader returns standardized dict: {'data', 'labels', 'subject_id'}

References:
  - DEAP: Koelstra et al., 2012 — 32 subjects, 32-ch EEG + 8 peripheral
  - DREAMER: Katsigiannis & Ramzan, 2018 — 23 subjects, 14-ch EEG + ECG
  - MAHNOB-HCI: Soleymani et al., 2012 — 30 subjects, 32-ch EEG + peripheral
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DEAPLoader:
    """
    DEAP Dataset Loader
    ────────────────────
    Load .dat files from data_preprocessed_python/
    Data shape per subject: (40 trials × 40 channels × 8064 time points)
      - Channels 0-31: EEG, Channels 32-39: Peripheral (GSR, ECG, EOG, EMG, temp, resp)
    Labels: arousal, valence (1-9 scale), binarized at threshold 5
    Anxiety: high arousal (≥5) + low valence (<5) → anxiety=1, else=0
    For 3-class: Low (low arousal), Moderate (high arousal + high valence), High (high arousal + low valence)
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.n_subjects = 32
        self.n_eeg_channels = 32
        self.n_peripheral_channels = 8
        self.n_trials = 40
        self.sampling_rate = 128  # preprocessed version is at 128 Hz
        self.n_timepoints = 8064
        
        self.eeg_channel_names = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
            'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz'
        ]
    
    def load_subject(self, subject_id: int) -> Dict[str, Any]:
        """
        Load data for a single subject.
        
        Args:
            subject_id: Subject number (1-32)
            
        Returns:
            dict with 'eeg_data' (40, 32, 8064), 'peripheral_data' (40, 8, 8064),
            'labels' (40,), 'arousal' (40,), 'valence' (40,), 'subject_id'
        """
        filename = f"s{subject_id:02d}.dat"
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"DEAP data file not found: {filepath}\n"
                f"Download from: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html\n"
                f"Place .dat files in: {self.data_dir}"
            )
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # data['data'] shape: (40, 40, 8064) — trials × channels × samples
        # data['labels'] shape: (40, 4) — trials × [valence, arousal, dominance, liking]
        signals = data['data']
        labels_raw = data['labels']
        
        eeg_data = signals[:, :self.n_eeg_channels, :]         # (40, 32, 8064)
        peripheral_data = signals[:, self.n_eeg_channels:, :]  # (40, 8, 8064)
        
        valence = labels_raw[:, 0]   # 1-9
        arousal = labels_raw[:, 1]   # 1-9
        
        # 3-class anxiety labeling:
        # High anxiety: high arousal (≥5) + low valence (<5)
        # Moderate: high arousal (≥5) + high valence (≥5) OR low arousal (<5) + low valence (<5)
        # Low anxiety: low arousal (<5) + high valence (≥5)
        anxiety_labels = np.zeros(self.n_trials, dtype=int)
        for i in range(self.n_trials):
            if arousal[i] >= 5 and valence[i] < 5:
                anxiety_labels[i] = 2  # High
            elif arousal[i] < 5 and valence[i] >= 5:
                anxiety_labels[i] = 0  # Low
            else:
                anxiety_labels[i] = 1  # Moderate
        
        logger.info(f"Loaded DEAP subject {subject_id}: {eeg_data.shape}, labels={np.bincount(anxiety_labels)}")
        
        return {
            'eeg_data': eeg_data,
            'peripheral_data': peripheral_data,
            'labels': anxiety_labels,
            'arousal': arousal,
            'valence': valence,
            'subject_id': subject_id,
            'channel_names': self.eeg_channel_names,
            'sampling_rate': self.sampling_rate
        }
    
    def load_all(self) -> Dict[str, Any]:
        """Load all 32 subjects and concatenate."""
        all_eeg, all_labels, all_subjects = [], [], []
        
        for sid in range(1, self.n_subjects + 1):
            try:
                subj = self.load_subject(sid)
                all_eeg.append(subj['eeg_data'])
                all_labels.append(subj['labels'])
                all_subjects.extend([sid] * self.n_trials)
            except FileNotFoundError as e:
                logger.warning(f"Skipping subject {sid}: {e}")
        
        if not all_eeg:
            raise RuntimeError("No DEAP data files found.")
        
        return {
            'data': np.concatenate(all_eeg, axis=0),
            'labels': np.concatenate(all_labels, axis=0),
            'subject_ids': np.array(all_subjects),
            'channel_names': self.eeg_channel_names,
            'sampling_rate': self.sampling_rate
        }
    
    def is_available(self) -> bool:
        """Check if at least one subject file exists."""
        return any((self.data_dir / f"s{i:02d}.dat").exists() for i in range(1, 33))


class DREAMERLoader:
    """
    DREAMER Dataset Loader
    ───────────────────────
    Load DREAMER.mat file.
    14-channel EEG (Emotiv EPOC) + 2-channel ECG.
    23 subjects × 18 film clips.
    Arousal/valence/dominance on scale 1-5, binarized at 2.5.
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.n_subjects = 23
        self.n_trials = 18
        self.n_eeg_channels = 14
        self.sampling_rate = 128
        
        self.eeg_channel_names = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
            'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
        ]
    
    def load(self) -> Dict[str, Any]:
        """
        Load the entire DREAMER dataset.
        
        Returns:
            dict with 'data', 'labels', 'subject_ids'
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"DREAMER.mat not found at: {self.data_path}\n"
                f"Download from: https://zenodo.org/record/546113\n"
                f"Place DREAMER.mat in the expected directory."
            )
        
        # Try mat73 first (MATLAB v7.3+), then scipy
        try:
            import mat73
            mat_data = mat73.loadmat(str(self.data_path))
        except Exception:
            from scipy.io import loadmat
            mat_data = loadmat(str(self.data_path), squeeze_me=True)
        
        dreamer = mat_data['DREAMER']
        all_eeg, all_labels, all_subjects = [], [], []
        
        for subj_idx in range(self.n_subjects):
            try:
                subj_data = dreamer['Data'][subj_idx]
                
                for trial_idx in range(self.n_trials):
                    # Extract EEG data
                    eeg = subj_data['EEG']['stimuli'][trial_idx]
                    
                    if isinstance(eeg, np.ndarray) and eeg.ndim == 2:
                        # Ensure shape is (channels, samples)
                        if eeg.shape[0] > eeg.shape[1]:
                            eeg = eeg.T
                        
                        all_eeg.append(eeg[:self.n_eeg_channels, :])
                        
                        # Extract labels
                        valence = float(subj_data['ScoreValence'][trial_idx])
                        arousal = float(subj_data['ScoreArousal'][trial_idx])
                        
                        # 3-class anxiety labeling (same logic as DEAP but scale 1-5)
                        threshold = 2.5
                        if arousal >= threshold and valence < threshold:
                            label = 2  # High anxiety
                        elif arousal < threshold and valence >= threshold:
                            label = 0  # Low anxiety
                        else:
                            label = 1  # Moderate
                        
                        all_labels.append(label)
                        all_subjects.append(subj_idx + 1)
                        
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Error loading DREAMER subject {subj_idx + 1}, trial {trial_idx}: {e}")
                continue
        
        if not all_eeg:
            raise RuntimeError("Failed to load any data from DREAMER.mat")
        
        logger.info(f"Loaded DREAMER: {len(all_eeg)} trials from {len(set(all_subjects))} subjects")
        
        return {
            'data': all_eeg,  # List of variable-length arrays
            'labels': np.array(all_labels),
            'subject_ids': np.array(all_subjects),
            'channel_names': self.eeg_channel_names,
            'sampling_rate': self.sampling_rate
        }
    
    def is_available(self) -> bool:
        return self.data_path.exists()


class MAHNOBLoader:
    """
    MAHNOB-HCI Dataset Loader
    ──────────────────────────
    Parse session folders with .bdf (BioSemi) + .xml annotation files.
    32-channel EEG + peripheral signals.
    30 subjects, ~20 trials each.
    Arousal/valence labels from XML (1-9 scale).
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.n_eeg_channels = 32
        self.target_sampling_rate = 256
        
        self.eeg_channel_names = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
            'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz'
        ]
    
    def _parse_xml_labels(self, xml_path: Path) -> Optional[Dict[str, float]]:
        """Parse XML annotation file for arousal/valence labels."""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            arousal = None
            valence = None
            
            for felt in root.iter('feltArsl'):
                arousal = float(felt.text)
            for felt in root.iter('feltVlnc'):
                valence = float(felt.text)
            
            if arousal is None or valence is None:
                # Try alternative XML structure
                for tag in root.iter():
                    if 'arousal' in tag.tag.lower():
                        arousal = float(tag.text) if tag.text else None
                    if 'valence' in tag.tag.lower():
                        valence = float(tag.text) if tag.text else None
            
            if arousal is not None and valence is not None:
                return {'arousal': arousal, 'valence': valence}
            
        except Exception as e:
            logger.warning(f"Failed to parse XML {xml_path}: {e}")
        
        return None
    
    def load(self) -> Dict[str, Any]:
        """
        Load all sessions from MAHNOB-HCI.
        
        Returns:
            dict with 'data', 'labels', 'subject_ids'
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MAHNOB-HCI directory not found: {self.data_dir}\n"
                f"Download from: https://mahnob-db.eu/hci-tagging/\n"
                f"Extract session folders into: {self.data_dir}"
            )
        
        import mne
        
        all_eeg, all_labels, all_subjects = [], [], []
        
        session_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        for session_dir in session_dirs:
            bdf_files = list(session_dir.glob("*.bdf"))
            xml_files = list(session_dir.glob("*.xml"))
            
            if not bdf_files or not xml_files:
                continue
            
            bdf_path = bdf_files[0]
            xml_path = xml_files[0]
            
            # Parse labels
            labels = self._parse_xml_labels(xml_path)
            if labels is None:
                continue
            
            try:
                # Read BDF file with MNE
                raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
                
                # Pick EEG channels
                eeg_picks = mne.pick_types(raw.info, eeg=True)[:self.n_eeg_channels]
                eeg_data = raw.get_data(picks=eeg_picks)
                
                # Downsample to target rate
                if raw.info['sfreq'] != self.target_sampling_rate:
                    from scipy.signal import resample
                    n_target = int(eeg_data.shape[1] * self.target_sampling_rate / raw.info['sfreq'])
                    eeg_data = resample(eeg_data, n_target, axis=1)
                
                # Anxiety labeling
                arousal = labels['arousal']
                valence = labels['valence']
                threshold = 5
                
                if arousal >= threshold and valence < threshold:
                    anxiety_label = 2  # High
                elif arousal < threshold and valence >= threshold:
                    anxiety_label = 0  # Low
                else:
                    anxiety_label = 1  # Moderate
                
                all_eeg.append(eeg_data)
                all_labels.append(anxiety_label)
                
                # Extract subject ID from folder name
                try:
                    subj_id = int(session_dir.name.split('_')[0].replace('Part', '').replace('P', ''))
                except ValueError:
                    subj_id = hash(session_dir.name) % 100
                
                all_subjects.append(subj_id)
                
            except Exception as e:
                logger.warning(f"Failed to load session {session_dir.name}: {e}")
                continue
        
        if not all_eeg:
            raise RuntimeError("No MAHNOB-HCI sessions loaded successfully.")
        
        logger.info(f"Loaded MAHNOB-HCI: {len(all_eeg)} sessions from {len(set(all_subjects))} subjects")
        
        return {
            'data': all_eeg,
            'labels': np.array(all_labels),
            'subject_ids': np.array(all_subjects),
            'channel_names': self.eeg_channel_names,
            'sampling_rate': self.target_sampling_rate
        }
    
    def is_available(self) -> bool:
        if not self.data_dir.exists():
            return False
        return any(self.data_dir.iterdir())


def get_loader(dataset_name: str):
    """Factory function to get the appropriate dataset loader."""
    from config import DEAP_DIR, DREAMER_PATH, MAHNOB_DIR
    
    loaders = {
        'DEAP': lambda: DEAPLoader(str(DEAP_DIR)),
        'DREAMER': lambda: DREAMERLoader(str(DREAMER_PATH)),
        'MAHNOB': lambda: MAHNOBLoader(str(MAHNOB_DIR))
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name]()
