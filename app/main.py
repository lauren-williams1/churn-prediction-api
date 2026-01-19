"""
FASTAPI APPLICATION - CHURN PREDICTION API
===========================================

Purpose:
    RESTful API server that loads the trained churn prediction model
    and serves predictions via HTTP endpoints. Provides model health
    checks and metadata information.

Architecture:
    - FastAPI framework for async HTTP server
    - Loads model artifacts on startup (not per request)
    - Validates inputs (will add Pydantic models tomorrow)
    - Returns JSON responses

Input (from model artifacts):
    - models/model.pkl: Trained RandomForestClassifier
    - models/label_encoders.pkl: Encoders for categorical features
    - models/feature_names.pkl: Feature names in training order
    - models/metadata.pkl: Model performance metrics

Endpoints:
    GET  /           → API welcome message with endpoint list
    GET  /health     → Health check (is model loaded?)
    GET  /model/info → Model metadata (accuracy, features, type)
    POST /predict    → Make prediction (TO BE ADDED TOMORROW)

Server Configuration:
    - Host: localhost (127.0.0.1)
    - Port: 8000 (default)
    - Reload: True (auto-restart on code changes)
    - Docs: /docs (Swagger UI)
    - Alternative docs: /redoc (ReDoc UI)

Connection to Other Files:
    ← Loads: models/*.pkl (created by train_model.py)
    → Serves: HTTP API (consumed by clients/frontend)
    
Startup Process:
    1. FastAPI initializes app
    2. @on_event("startup") triggers load_model()
    3. Model artifacts loaded into global variables
    4. Server ready to accept requests

Global Variables:
    - model: RandomForestClassifier instance
    - encoders: Dict of LabelEncoders for categorical features
    - features: List of feature names in correct order

Error Handling:
    - Returns 503 if model not loaded
    - Returns 422 for invalid input (Pydantic validation)
    - Returns 500 for prediction errors

When to Run:
    - During development (with --reload flag)
    - In production (with gunicorn/uvicorn workers)

Example Usage:
    # Development
    $ python3 -m uvicorn app.main:app --reload
    
    # Production
    $ uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

Testing:
    # Browser
    http://localhost:8000/docs
    
    # curl
    $ curl http://localhost:8000/health
    {"status":"healthy","model_loaded":true}

Production Considerations (from Chip Huyen):
    - Load model once on startup (not per request) ✅
    - Use async endpoints for I/O operations (future)
    - Add request logging (future)
    - Add monitoring/metrics (future)
    - Consider batch prediction endpoint (future)
"""

from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Global variables to store model artifacts
# Loaded once on startup, reused for all requests
model = None
encoders = None
features = None

@app.on_event("startup")
async def load_model():
    """
    Load model artifacts when server starts.
    
    This runs ONCE when the server starts, not on every request.
    Loading once improves latency (no disk I/O per request).
    
    Loaded Artifacts:
        - model: RandomForestClassifier for predictions
        - encoders: LabelEncoders to transform categorical inputs
        - features: Feature names to ensure correct column order
    
    Raises:
        FileNotFoundError: If model files don't exist (run train_model.py first)
    """
    global model, encoders, features
    
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    
    print("✅ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Features: {len(features)}")

@app.get("/")
def root():
    """
    Root endpoint - API information.
    
    Returns:
        dict: Welcome message with available endpoints
        
    Example Response:
        {
            "message": "Churn Prediction API",
            "endpoints": {
                "health": "/health",
                "docs": "/docs"
            }
        }
    """
    return {
        "message": "Churn Prediction API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    """
    Health check endpoint.
    
    Used by:
        - Load balancers to check if service is up
        - Monitoring systems (Prometheus, Datadog, etc.)
        - Deployment pipelines (readiness probes)
    
    Returns:
        dict: Health status and model load state
        
    Example Response:
        {
            "status": "healthy",
            "model_loaded": true
        }
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/model/info")
def model_info():
    """
    Get model metadata and performance metrics.
    
    Useful for:
        - Debugging which model version is deployed
        - Monitoring model performance over time
        - Comparing deployed model to training results
    
    Returns:
        dict: Model metadata including:
            - accuracy: Test set accuracy
            - roc_auc: ROC-AUC score
            - features: List of feature names
            - model_type: Type of model (RandomForest, etc.)
    
    Example Response:
        {
            "accuracy": 0.805,
            "roc_auc": 0.847,
            "features": ["gender", "SeniorCitizen", ...],
            "model_type": "RandomForest"
        }
    
    Raises:
        FileNotFoundError: If metadata.pkl doesn't exist
    """
    with open('models/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return metadata