# Data Engineering & Machine Learning Operations Report
**Project Designation:** Neuriq AI
**Primary Focus:** Brainwave-based Anxiety Detection Pipeline

This document details the architectural decisions, pipeline workflows, and development contributions handled by the Data/ML Engineering branch of the project. It covers data acquisition parsing, signal preprocessing, feature extraction, model engineering, and deployment mechanisms.

---

## 1. Data Collection & Ingestion Pipelines

The data ingestion module ([ml_service/data/loaders.py](file:///d:/Work/Projects/Neuriq/ml_service/data/loaders.py)) was engineered to normalize and structure complex, raw physiological signals from three distinct academic emotion-recognition datasets. The primary challenge resolved was establishing unified signal representations, uniform lengths, and standardized target classes across completely disparate file formats.

### Supported Datasets

1. **DEAP Dataset Loader**
   - **Characteristics**: 32 Subjects, 40 trial clips mapping to 32 EEG channels at 128Hz. Contains additional peripheral signals.
   - **Pipeline Task**: Loaded directly from pickled binary chunks (`.dat`). 
   - **Target Labeling**: Merged Valence and Arousal variables (from 1-9 rating scales) to formulate a precise 3-tier Anxiety representation: High (Arousal $\geq$ 5, Valence < 5), Moderate, and Low anxiety threshold classifications.

2. **DREAMER Dataset Parser**
   - **Characteristics**: 23 Subjects executing 18 distinct experimental trials. 14 EEG channels harvested from Emotiv EPOC hardware plus ECG elements.
   - **Pipeline Task**: Deployed a dynamic MATLAB object scanner using `mat73`/`scipy.io` to interpret deeply nested `.mat` hierarchies mapping exactly down to the trial-timestamp layers.

3. **MAHNOB-HCI Engine**
   - **Characteristics**: 30 Subjects utilizing high-density 32-channel BioSemi machines recording continuously in real-time.
   - **Pipeline Task**: Orchestrated an MNE-Python extraction pipeline traversing directories iteratively to read raw continuous `.bdf` blocks, cross-referencing emotional metadata parsed from independently saved `.xml` annotation trees. Signals were structurally downsampled dynamically to maintain universal standard frequencies.

---

## 2. Signal Preprocessing & Feature Extraction

Raw brainwaves are natively incredibly noisy. To optimize dataset stability before modeling, the pipeline extracts specialized engineered EEG metrics through an automated `features/` directory engine.

* **Frequency Domain Operations ([frequency_domain.py](file:///d:/Work/Projects/Neuriq/ml_service/features/frequency_domain.py))**: Responsible for filtering signals into specific standardized brain topography rhythms (Delta, Theta, Alpha, Beta, Gamma) utilizing Power Spectral Density (Welch's Method) to examine cognitive load ratios (such as Alpha/Beta asymmetric markers known to correlate with negative neurological affect).
* **Time Domain Subroutines ([time_domain.py](file:///d:/Work/Projects/Neuriq/ml_service/features/time_domain.py))**: Extractor logic responsible for flattening waves via statistical moments—Variance, Moving Averages, Skewness, Kurtosis.
* **Non-Linear Dynamics ([nonlinear.py](file:///d:/Work/Projects/Neuriq/ml_service/features/nonlinear.py))**: Computation of complex entropy metrics and fractal dimension modeling to measure the "unpredictability" patterns characteristic of high-arousal stress incidents.

---

## 3. Modeling Architectures & Ensembles

We scaled the prediction ecosystem to test both highly transparent, classical techniques as well as state-of-the-art Deep Learning models (`ml_service/models/`).

### Implemented Model Classes

* **Advanced Boosting Clusters (`boosting.py`)**: Incorporates XGBoost, LightGBM, and CatBoost algorithms. These implementations include embedded Optuna hyperparameter tuning routines capable of dynamically adjusting trees to offset class imbalances organically.
* **Classical Baselines (`classical.py` & `ensemble.py`)**: Incorporates structured Support Vector Machines (SVMs) and Random Forests connected through a Meta-Logistic Regression top-layer.
* **Deep Neural Configurations (`models/deep/`)**: Implementation architectures for pure spatial-temporal evaluations, featuring frameworks matching recent academic benchmarks:
  - **Brain2Vec & CNN-LSTM Pipelines**: Fuses multi-channel Convolutional layers mapping topographies into recurrent Long Short-Term Memory logic, isolating temporal anxiety spikes.
  - **EEGNet**: Incorporates compact depthwise convolutions specifically modeled to track minute microvolt variations inside raw neuro-signals without overfitting.

---

## 4. API Backend & Service Connectivity

A high-performance Python application was forged utilizing **FastAPI** (`ml_service/main.py`) to expose the underlying deep learning logic seamlessly to web applications.

* **Asynchronous Training Mechanisms**: Exposes a `POST /train` endpoint allowing the frontend application to instigate massive model evaluations on demand. Because model generation is slow, it fires `BackgroundTasks`, simulating internal logging outputs non-blockingly while holding server availability open.
* **Inference Endpoint Handling**: Incorporates explicit validation models utilizing `Pydantic` mapping signal-matrices (`POST /predict`) reliably directly into prediction subroutines.
* **Continuous Integration Ready**: Built alongside comprehensive environment scripts (`requirements.txt`) mapped seamlessly to NextJS application endpoints, eliminating deployment configuration friction.
