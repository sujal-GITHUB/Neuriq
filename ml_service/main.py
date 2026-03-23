"""
NeuroAnxiety ML Service — FastAPI Application
==============================================
Main entry point for the Python ML microservice.
Exposes endpoints for prediction, training, metrics, and data management.
"""

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import time
import asyncio

from config import CORS_ORIGINS, API_HOST, API_PORT
from mock_responses import (
    mock_prediction_response,
    mock_training_response,
    mock_training_status,
    mock_metrics_response,
    mock_models_list,
    mock_datasets_info,
    mock_upload_response
)

# ── FastAPI App ─────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroAnxiety ML Service",
    description="EEG-based mental anxiety detection using ML/DL models",
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

# ── In-memory training job store ────────────────────────────────────
training_jobs: Dict[str, Dict[str, Any]] = {}


# ── Request/Response Models ─────────────────────────────────────────
class PredictRequest(BaseModel):
    signals: List[List[float]] = Field(..., description="2D array: [channels][samples]")
    channels: List[str] = Field(..., description="Channel names")
    sampling_rate: int = Field(128, description="Sampling rate in Hz")
    modalities: List[str] = Field(default=["EEG"], description="Signal modalities")
    model: str = Field(default="brain2vec", description="Model to use for inference")


class ManualPredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Named feature values")
    model: str = Field(default="brain2vec", description="Model to use")


class TrainRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name: DEAP, DREAMER, or MAHNOB")
    model: str = Field(..., description="Model: ensemble, brain2vec, cnn_lstm, eegnet")
    epochs: int = Field(default=100)
    batch_size: int = Field(default=64)
    learning_rate: float = Field(default=1e-4)
    cv_folds: int = Field(default=5)
    use_smote: bool = Field(default=True)
    eval_split: str = Field(default="kfold", description="kfold or loso")


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "NeuroAnxiety ML Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """Run anxiety prediction on provided EEG signals."""
    start_time = time.time()
    
    # TODO: Replace with real inference in Step 12
    result = mock_prediction_response(request.model)
    result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return result


@app.post("/predict/manual")
async def predict_manual(request: ManualPredictRequest):
    """Run anxiety prediction from manually entered features."""
    start_time = time.time()
    
    # TODO: Replace with real inference in Step 12
    result = mock_prediction_response(request.model)
    result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return result


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Start a training job asynchronously."""
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
    
    # TODO: Replace with real training in Step 12
    background_tasks.add_task(mock_training_loop, job_id, request.epochs)
    
    return {"status": "started", "job_id": job_id}


async def mock_training_loop(job_id: str, total_epochs: int):
    """Simulate training progress."""
    import random
    for epoch in range(1, total_epochs + 1):
        await asyncio.sleep(0.5)  # Simulate training time
        progress = (epoch / total_epochs) * 100
        training_jobs[job_id] = {
            "status": "running" if epoch < total_epochs else "completed",
            "progress": round(progress, 1),
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "current_metrics": {
                "train_loss": round(1.2 - (progress / 100) * 0.9 + random.uniform(-0.05, 0.05), 4),
                "val_loss": round(1.3 - (progress / 100) * 0.8 + random.uniform(-0.08, 0.08), 4),
                "train_accuracy": round(0.35 + (progress / 100) * 0.55 + random.uniform(-0.03, 0.03), 4),
                "val_accuracy": round(0.30 + (progress / 100) * 0.50 + random.uniform(-0.05, 0.05), 4),
                "learning_rate": 1e-4 * (1 - progress / 200)
            },
            "log": [f"Epoch {epoch}/{total_epochs} - loss: {round(1.2 - (progress/100)*0.9, 4)} - acc: {round(0.35 + (progress/100)*0.55, 4)}"]
        }


@app.get("/training-status/{job_id}")
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    if job_id in training_jobs:
        return training_jobs[job_id]
    
    # Return mock status for unknown job IDs (development convenience)
    return mock_training_status()


@app.get("/metrics/{model_name}/{dataset}")
async def get_metrics(model_name: str, dataset: str):
    """Get full metrics for a trained model on a specific dataset."""
    # TODO: Replace with real metrics from saved results in Step 12
    return mock_metrics_response(model_name, dataset)


@app.get("/models")
async def list_models():
    """List all available trained model checkpoints."""
    # TODO: Replace with real model discovery in Step 12
    return mock_models_list()


@app.get("/datasets")
async def list_datasets():
    """Get info about available datasets."""
    # TODO: Replace with real dataset scanning in Step 12
    return mock_datasets_info()


@app.post("/upload-eeg")
async def upload_eeg(file: UploadFile = File(...)):
    """Upload and parse an EEG file (.csv, .edf, .bdf, .dat)."""
    allowed_extensions = {'.csv', '.edf', '.bdf', '.dat'}
    
    if file.filename:
        ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(allowed_extensions)}"
            )
    
    # TODO: Replace with real file parsing in Step 12
    result = mock_upload_response(file.filename or "unknown.csv")
    
    return result


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
