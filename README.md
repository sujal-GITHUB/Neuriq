# Neuriq: Clinical EEG-Based Anxiety & Stress Detection

Neuriq is an end-to-end machine learning platform designed to classify psychiatric anxiety and stress levels from multi-channel electroencephalogram (EEG) brainwave signals. It combines a Python FastAPI ML microservice with a Next.js frontend web application.

---

## 1. Project Architecture & Codebase Structure

The project is structured into two main components: `ml_service` (the Python FastAPI backend) and `website` (the Next.js React frontend).

```
Neuriq/
├── ml_service/                  # FastAPI Machine Learning Service
│   ├── data/
│   │   ├── dataset_loaders.py   # Loader & re-mapper for BRMH, DASPS, and SEED datasets
│   │   └── eeg_preprocessor.py  # Z-score outlier filtration, imputation, & scaling
│   ├── features/
│   │   ├── eeg_feature_engineering.py # Clinically validated band ratios, asymmetry, and averages
│   │   └── eeg_pca_rfe.py       # Legacy PCA-RFE hybrid feature selection
│   ├── models/
│   │   ├── checkpoints/         # Serialized .joblib trained checkpoints
│   │   └── xgforest_stacking.py # Stacking ensemble training pipeline (LDA, RF, ExtraTrees, XGBoost)
│   ├── evaluation/
│   │   ├── metrics.py           # Per-class specificity, McNemar & Friedman tests, CV summaries
│   │   └── explainability.py    # SHAP-based feature importance analyzer
│   ├── RAG/                     # Academic reference literature (PDFs)
│   ├── results/
│   │   └── eeg_model_evaluation.json # Exported evaluation JSON metrics
│   ├── scratch/                 # Experimental scripts & sweeps
│   ├── config.py                # Hyperparameters, channel structures, and paths
│   ├── main.py                  # FastAPI server entry point
│   └── requirements.txt         # Backend Python dependencies
│
└── website/                     # Next.js Frontend Application
    ├── src/
    │   ├── app/
    │   │   ├── analyze/         # Live signal loading, prediction, and visualizer page
    │   │   ├── assistant/       # RAG assistant chat page (proxied to OpenRouter)
    │   │   ├── models/          # Model parameters configuration and checkpoint catalog
    │   │   ├── research/        # Academic timelines, 10-20 electrode map, and glossary
    │   │   ├── train/           # Training studio interface with live progress monitoring
    │   │   └── api/             # API handlers proxying frontend requests to FastAPI backend
    │   ├── components/          # Reusable layouts, Recharts components, and UI components
    │   ├── store/               # Zustand stores for state management
    │   └── lib/                 # Shared utilities and API client wrapper
    ├── tailwind.config.ts       # Tailwind CSS configuration
    └── package.json             # Frontend package configurations and scripts
```

---

## 2. Ingested Datasets

Neuriq harmonizes three academic emotion-recognition and psychiatric datasets:

1. **BRMH (Kaggle Brain-Related Mental Health)**:
   - **Characteristics**: Tabular CSV file containing 330 samples (Healthy, Anxiety, and Trauma/Stress).
   - **Input Space**: Over 1,500 columns representing pre-computed Absolute Powers (AB) and Coherence (COH) pairs.
   - **Topography**: 19 standard channels mapped using the standard 10-20 system.

2. **DASPS (Anxious States based on Psychological Stimulation)**:
   - **Characteristics**: Matlab format (`.mat`) containing raw EEG recording trials from 23 subjects (276 total situation trials).
   - **Input Space**: 14 channels (Emotiv EPOC layout) recorded at 128Hz.
   - **Processing**: Welch Power Spectral Density (PSD) calculation is mapped into standard 10-20 positions; unmapped channels are zero-padded to share the same feature width as BRMH.

3. **SEED (SJTU Emotion EEG Dataset)**:
   - **Characteristics**: Binary NPZ chunks containing 50,910 samples.
   - **Input Space**: 62 channels across 5 frequency bands.
   - **Remapping**: Emotion targets are mapped to anxiety proxies (Negative affect → Acute Stress, Neutral/Positive → Healthy) for auxiliary training augmentation.

---

## 3. Data & Feature Engineering Pipeline

### Preprocessing & Cleaning (`data/eeg_preprocessor.py`)
* **Duplicate Removal**: Filters out repeated records to prevent evaluation leakage.
* **Imputation**: Replaces null entries with average column statistics.
* **Outlier Filtration**: Calculates sample Z-scores and clamps outliers exceeding $3\sigma$ boundaries.
* **Scaling**: Translates values to standard normal distributions via `StandardScaler` (with a fallback to `MinMaxScaler` in legacy modes).

### Clinically Validated Features (`features/eeg_feature_engineering.py`)
* **Relative Band Power (RP)**: Evaluated across Delta, Theta, Alpha, Beta, High-Beta, and Gamma bands per channel.
* **Band Ratios**:
  - *Theta/Beta Ratio (TBR)*: Clinically validated marker for ADHD and cognitive focus.
  - *Theta/Alpha (TAR), Beta/Alpha (BAR), Delta/Alpha (DAR)*, and *(Theta+Beta)/Alpha (TBAR)*.
* **Hemispheric Asymmetry**: Log-ratios of homologous channels (e.g., FP1/FP2, F3/F4) across all spectral bands.
* **Frontal Alpha Asymmetry (FAA)**: Frontal asymmetry indices ($F_4 - F_3$) serving as primary indicators of anxiety and stress.
* **Regional & Global Averages**: Statistical aggregation (mean, std, max) over Frontal, Central, Temporal, Parietal, and Occipital zones.

---

## 4. Machine Learning & Ensemble Stacking

To handle high-dimensional collinear features and maximize generalization, Neuriq applies a two-stage classification workflow:

### Feature Selection
* **SelectKBest Selection**: Applies ANOVA F-tests (`f_classif`) to reduce the 1,521-dimensional input feature space down to the top 50 highly discriminative indices.
* **SMOTE Oversampling**: Resolves class imbalances by synthesizing minority class vectors to balance class distributions.

### Stacking Classifier (`models/xgforest_stacking.py`)
* **Base Classifiers**:
  1. *Linear Discriminant Analysis (LDA)*: Linear classification with automatic shrinkage.
  2. *Random Forest Classifier*: 300 estimators, max depth 7, balanced class weights.
  3. *Extra Trees Classifier*: 300 estimators, max depth 7, balanced class weights.
  4. *XGBoost Classifier*: 200 trees, learning rate 0.05, trained on CUDA GPU (when available) with automatic refitting to CPU to ensure local inference compatibility.
* **Meta-Classifier**: Logistic Regression with L2 regularization ($C=1.0$) trained via 5-Fold Cross Validation.

---

## 5. Performance Metrics & Results

The evaluation results of the current XGForest+ stacking pipeline on the real clinical BRMH dataset are recorded in `ml_service/results/eeg_model_evaluation.json`:

* **Classification Accuracy**: **64.05%** (a substantial improvement over the baseline 42.4% accuracy).
* **Macro Precision**: **58.09%**
* **Macro Recall**: **59.48%**
* **Macro F1-Score**: **58.50%**
* **Area Under ROC Curve (AUC)**:
  - Healthy Controls (Baseline): **0.8026**
  - Social Anxiety Disorder: **0.7987**
  - Acute Stress State: **0.7836**

### Top Contributory Bands
* **Beta Rhythms**: **90.23%** (the primary indicator of high cortical arousal).
* **Delta Rhythms**: **7.05%**
* **Alpha Rhythms**: **2.72%**

---

## 6. FastAPI Backend Endpoints

Exposed on port `8000` by default:

* `GET /`: Health check and model load status.
* `POST /predict`: Classifies raw EEG signal matrices. Expects raw signals: `[channels][samples]`.
* `POST /predict/manual`: Classifies named feature value dictionaries.
* `POST /train`: Asynchronously triggers the full training pipeline using `BackgroundTasks`.
* `GET /training-status/{job_id}`: Retrieves progress logs and metrics of a training task.
* `GET /metrics/{model}/{dataset}`: Retrieves the historical model evaluation summary.
* `GET /models`: Lists available `.joblib` model checkpoints.
* `GET /datasets`: Lists active `.csv` dataset configurations.
* `POST /api/assistant/chat`: Clinician chat assistant proxying queries to OpenRouter models (configured via `.env`).

---

## 7. Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js 18+
- CUDA Toolkit (optional, for GPU-accelerated training)

### Step 1: Run the ML Microservice
1. Navigate to the microservice directory:
   ```bash
   cd ml_service
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables in `.env` (optional, for chat assistant):
   ```env
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_MODEL=openrouter/owl-alpha
   ```
5. Run the server:
   ```bash
   python main.py
   ```
   The API will be available at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.

### Step 2: Run the Next.js Frontend
1. Navigate to the website directory:
   ```bash
   cd website
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000` in your browser.
