"""
NeuroAnxiety — EEG Preprocessing Pipeline (Step 2)
==================================================
Implements three strict data-cleaning procedures:
  1. Data Imputation: Replaces missing values with column average values.
  2. Duplicate Removal: Removes duplicate rows from pandas DataFrame to eliminate prediction bias.
  3. Outlier Filtration: Calculates Z-score for each data point and clamps values exceeding Z=3.
  4. Min-Max Scaling: Maps all spectral features to a scale-free [0, 1] range.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Union

class EEGPreprocessor:
    """Preprocesses highly dimensional brainwave frequency bands according to clinical standards."""
    
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.column_means_ = {}
        self.feature_min_ = {}
        self.feature_max_ = {}
        self.column_stds_ = {}
        
    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> 'EEGPreprocessor':
        """Computes statistical metrics (mean, std, min, max) for scaling and imputation."""
        # Clean duplicates first
        df_clean = df.drop_duplicates()
        
        for col in feature_cols:
            vals = df_clean[col].values
            # Compute mean ignoring NaNs
            mean_val = float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else 0.0
            std_val = float(np.nanstd(vals)) if not np.all(np.isnan(vals)) else 1.0
            std_val = std_val if std_val > 1e-8 else 1.0
            
            # Temporary imputation to calculate outlier-free min/max
            imputed = np.where(np.isnan(vals), mean_val, vals)
            # Cap outliers temporarily at 3-sigma
            z = (imputed - mean_val) / std_val
            clipped = np.where(np.abs(z) > self.z_threshold, mean_val + np.sign(z) * self.z_threshold * std_val, imputed)
            
            self.column_means_[col] = mean_val
            self.column_stds_[col] = std_val
            self.feature_min_[col] = float(np.min(clipped))
            self.feature_max_[col] = float(np.max(clipped))
            
        return self
        
    def transform(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Applies data-cleaning, outlier filtration, and min-max scaling."""
        df_out = df.copy()
        
        # 1. Imputation (using column average values)
        for col in feature_cols:
            mean_val = self.column_means_.get(col, 0.0)
            df_out[col] = df_out[col].fillna(mean_val)
            
        # 2. Outlier Filtration (Z-score threshold = 3 clamping)
        for col in feature_cols:
            mean_val = self.column_means_.get(col, 0.0)
            std_val = self.column_stds_.get(col, 1.0)
            vals = df_out[col].values
            
            z_scores = (vals - mean_val) / std_val
            outliers = np.abs(z_scores) > self.z_threshold
            
            # Adjust outlier values to the exact 3-sigma boundary
            vals[outliers] = mean_val + np.sign(z_scores[outliers]) * self.z_threshold * std_val
            df_out[col] = vals
            
        # 3. Min-Max Scaling (maps features into a strict [0, 1] range)
        for col in feature_cols:
            min_val = self.feature_min_.get(col, 0.0)
            max_val = self.feature_max_.get(col, 1.0)
            diff = max_val - min_val
            if diff < 1e-8:
                df_out[col] = 0.0
            else:
                df_out[col] = (df_out[col] - min_val) / diff
                # Clip to exactly [0, 1] range to avoid floating-point drift
                df_out[col] = np.clip(df_out[col], 0.0, 1.0)
                
        return df_out

    def clean_and_preprocess_dataset(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Convenience method to execute full cleaning including duplicate removal."""
        # Drop duplicate rows
        df_cleaned = df.drop_duplicates()
        
        # Fit and Transform
        self.fit(df_cleaned, feature_cols)
        return self.transform(df_cleaned, feature_cols)
