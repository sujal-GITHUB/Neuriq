# Neuriq: Clinical EEG-Based Anxiety & Stress Detection

Neuriq is an end-to-end machine learning platform designed to classify psychiatric anxiety and stress levels from multi-channel electroencephalogram (EEG) brainwave signals. It combines a Python FastAPI ML microservice with a Next.js frontend web application.

---

## Revised Project Objectives

Based on the actual codebase implementation, the primary objectives of the project are:

### 1. Data Engineering & Preprocessing Pipeline
*   **Multi-Dataset Ingestion**: Develop an automated pipeline (`dataset_loaders.py`) capable of ingesting raw data from disparate formats:
    *   **BRMH (Kaggle Brain-Related Mental Health)**: Tabular CSV file (330 samples) with pre-computed absolute powers (AB) and coherences (COH).
    *   **DASPS**: Nested `.mat` MATLAB structs from 23 subjects (276 situations) mapped from Hamilton Anxiety Scale scores.
    *   **SEED**: Auxiliary NPZ binary chunks mapped to stress/healthy affect proxies.
*   **Preprocessing & Cleaning**: Apply automated NaN mean-imputation, $3\sigma$ (three-sigma) outlier clamping, and standard scaling (`StandardScaler`) to normalize feature distributions.
*   **Advanced Feature Engineering**: Extract clinically validated topographical and spectral features (`eeg_feature_engineering.py`) including:
    *   *Relative Band Powers* across Delta, Theta, Alpha, Beta, High-Beta, and Gamma bands.
    *   *Band Power Ratios* (Theta/Beta, Theta/Alpha, Beta/Alpha, Delta/Alpha) mapped per channel.
    *   *Hemispheric Asymmetry*: Log-ratio of homologous pairs (`ASYM_band_left_right`) and Frontal Alpha Asymmetry (`FAA_alpha` = F4 - F3) to evaluate cognitive load and emotional affect.
    *   *Regional & Global Statistics*: Mean, standard deviation, and maximum power across Frontal, Central, Temporal, Parietal, and Occipital zones.

### 2. Machine Learning Algorithm Implementation
*   **SelectKBest Feature Selection**: Apply ANOVA F-test (`f_classif`) to reduce the high-dimensional feature space (over 1,500 dimensions including AB and COH) to the top 50 highly discriminative features.
*   **Class Balancing**: Handle class distribution imbalances using Synthetic Minority Over-sampling Technique (SMOTE).
*   **XGForest+ Stacking Ensemble**: Train a robust ensemble classifier (`xgforest_stacking.py`) composed of four base models:
    *   *Linear Discriminant Analysis (LDA)* (with shrinkage)
    *   *Random Forest Classifier* (300 estimators)
    *   *Extra Trees Classifier* (300 estimators)
    *   *XGBoost Classifier* (trained on GPU via CUDA when available, with automatic fallback/refitting parameters to CPU for inference compatibility)
*   **Meta-Classification**: Connect the base predictions using a Logistic Regression meta-model trained via 5-Fold Cross Validation.
*   **Statistical Evaluation**: Calculate detailed performance metrics (Accuracy, Macro/Weighted F1, Cohen's Kappa, MCC, and ROC-AUC curves) and run statistical validation (McNemar's and Friedman tests).

### 3. Interactive Web Platform
*   **Next.js & Tailwind CSS Client**: Provide a responsive dashboard (`website/`) for loading files, running real-time inference, and tracking model metrics.
*   **FastAPI Backend Microservice**: Expose a non-blocking asynchronous training endpoint (`POST /train`) utilizing `BackgroundTasks` to train models on demand, and a prediction gateway (`POST /predict`) to run inferences.
*   **Dynamic API Proxying**: Implement Server-Side Next.js route handlers that forward frontend payload schemas to the FastAPI endpoints, translate probabilities into categorical levels (`Low`, `Moderate`, `High`, or `Both` for comorbidities), and map fallback mock responses during local development if the service is offline.

### 4. Visual Analytics & Explanations
*   **Recharts Data Visualizations**: Render real-time confidence scores, categorical probabilities, global band power distributions, and feature importances.
*   **Model Explainability**: Visualize performance via Confusion Matrices, ROC curves, and SHAP-value features to explain classifications and clinical decisions.
