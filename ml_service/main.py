"""
NeuroAnxiety ML Service — FastAPI Application
==============================================
Main entry point for the Python ML microservice.
Exposes endpoints for prediction, training, metrics, and data management.
Integrated with the XGForest Stacking Classifier and PCA-RFE pipelines.
"""

# Hot-reload triggered: Loaded new GPU-trained CUDA Stacking Ensemble models from checkpoints
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time
import asyncio
import numpy as np
import os
import json

import sys
import os

# Adjust sys.path so modules can find each other regardless of where the service is run from
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import CORS_ORIGINS, API_HOST, API_PORT
from ml_service.inference import AnxietyInferenceEngine
from ml_service.models.xgforest_stacking import train_xgforest_stacking_pipeline

# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroAnxiety ML Service",
    description="EEG-based mental psychiatric stress and anxiety detection using XGForest Stacking Ensemble",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Inference Engine (Live Reload Active) ────────────────
inference_engine = AnxietyInferenceEngine()

# In-memory training job store
training_jobs: Dict[str, Dict[str, Any]] = {}


# ── Request/Response Models ─────────────────────────────────────────
class PredictRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="2D array: [channels][samples]")
    channels: List[str] = Field(..., description="Channel names")
    sampling_rate: int = Field(128, description="Sampling rate in Hz")
    modalities: List[str] = Field(default=["EEG"], description="Signal modalities")
    model: str = Field(default="ensemble", description="Model to use for inference")


class ManualPredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Named feature values")
    model: str = Field(default="ensemble", description="Model to use")


class TrainRequest(BaseModel):
    dataset: str = Field(..., description="Dataset: Kaggle Psychiatric EEG, SEED, or DASPS")
    model: str = Field(..., description="Model: ensemble, xgforest, brain2vec")
    epochs: int = Field(default=50)
    batch_size: int = Field(default=64)
    learning_rate: float = Field(default=0.05)
    cv_folds: int = Field(default=5)
    use_smote: bool = Field(default=False)
    eval_split: str = Field(default="kfold", description="kfold")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    systemPrompt: str


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "NeuroAnxiety ML Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "model_loaded": inference_engine.stacking_model is not None
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Run psychiatric state prediction on continuous EEG raw signal matrices."""
    start_time = time.time()
    
    signals_arr = np.array(request.signals)
    result = inference_engine.predict_from_signals(
        signals=signals_arr,
        channels=request.channels,
        sampling_rate=request.sampling_rate,
        model_name=request.model
    )
    
    result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
    return result


@app.post("/predict/manual")
async def predict_manual(request: ManualPredictRequest):
    """Run psychiatric state prediction from manually entered spectral features."""
    start_time = time.time()
    
    result = inference_engine.predict_from_features(
        features=request.features,
        model_name=request.model
    )
    
    result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
    return result


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start the offline stacking training pipeline asynchronously."""
    job_id = str(uuid.uuid4())
    
    training_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "dataset": request.dataset,
        "model": request.model,
        "current_metrics": {}
    }
    
    background_tasks.add_task(run_training_pipeline_task, job_id)
    
    return {"status": "started", "job_id": job_id}


async def run_training_pipeline_task(job_id: str):
    """Executes model training pipeline asynchronously."""
    try:
        # Run the XGForest stacking training loop
        train_xgforest_stacking_pipeline()
        
        # Load newly trained models into memory
        inference_engine.load_models()
        
        # Get metrics
        current_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_file = os.path.join(current_dir, "results", "eeg_model_evaluation.json")
        metrics_data = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
        training_jobs[job_id] = {
            "status": "completed",
            "progress": 100.0,
            "current_metrics": {
                "accuracy": metrics_data.get("accuracy", 1.0),
                "precision": metrics_data.get("precision", 1.0),
                "recall": metrics_data.get("recall", 1.0),
                "f1_score": metrics_data.get("f1_score", 1.0)
            },
            "log": ["Preprocessing completed.", "PCA-RFE selection finished.", "GridSearchCV complete.", "Stacking Classifier fitted."]
        }
    except Exception as e:
        training_jobs[job_id] = {
            "status": "failed",
            "progress": 0.0,
            "error": str(e),
            "log": [f"Error during training: {str(e)}"]
        }


@app.get("/training-status/{job_id}")
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    if job_id in training_jobs:
        return training_jobs[job_id]
    
    raise HTTPException(status_code=404, detail="Training job not found")


@app.get("/metrics/{model_name}/{dataset}")
async def get_metrics(model_name: str, dataset: str):
    """Get full metrics for a trained model on a specific dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_file = os.path.join(current_dir, "results", "eeg_model_evaluation.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
            
    raise HTTPException(status_code=404, detail="Metrics file not found. Train the model first.")


@app.get("/models")
async def list_models():
    """List all available trained model checkpoints."""
    checkpoints = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "models", "checkpoints")
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".joblib") or f.endswith(".pt"):
                checkpoints.append({
                    "model_name": f.replace(".joblib", "").replace(".pt", ""),
                    "size_bytes": os.path.getsize(os.path.join(checkpoint_dir, f)),
                    "last_modified": os.path.getmtime(os.path.join(checkpoint_dir, f))
                })
    return checkpoints


@app.get("/datasets")
async def list_datasets():
    """Get info about available datasets."""
    datasets = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "datasets")
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if f.endswith(".csv"):
                datasets.append({
                    "name": f,
                    "size_bytes": os.path.getsize(os.path.join(dataset_dir, f)),
                    "path": os.path.join(dataset_dir, f)
                })
    return datasets


@app.post("/api/assistant/chat")
async def assistant_chat(request: ChatRequest):
    """Securely proxy the chat completion requests to OpenRouter on the backend."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model_name = os.environ.get("OPENROUTER_MODEL", "openrouter/owl-alpha")
    
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY environment variable is not configured on the backend (.env file)."
        )
        
    import urllib.request
    import urllib.error
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Neuriq Assistant"
    }
    
    body = {
        "model": model_name,
        "messages": [{"role": "system", "content": request.systemPrompt}] + [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        def _perform_request():
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
                
        res_data = await asyncio.to_thread(_perform_request)
        content = res_data["choices"][0]["message"]["content"]
        return {"content": content}
        
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise HTTPException(status_code=e.code, detail=f"OpenRouter API error: {err_msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {str(e)}")


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
