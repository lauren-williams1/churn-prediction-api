"""Model serving adapter.

Loads the 'champion' model from the MLflow registry when MLFLOW_TRACKING_URI is
set; otherwise falls back to the exported local pipeline (models/pipeline.joblib),
which is what the Docker image uses when no MLflow server is reachable.
"""
import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))
LOCAL_MODEL = Path(os.getenv("LOCAL_MODEL_PATH", "models/pipeline.joblib"))
LOCAL_META = Path(os.getenv("LOCAL_META_PATH", "models/metadata.json"))


class ModelService:
    def __init__(self, pipeline, version: str, source: str, threshold: float = THRESHOLD):
        self.pipeline = pipeline
        self.version = version
        self.source = source
        self.threshold = threshold

    def predict(self, features: dict) -> dict:
        """features: raw customer fields (same columns as the training CSV, minus Churn)."""
        df = pd.DataFrame([features])
        proba = float(self.pipeline.predict_proba(df)[0, 1])
        return {
            "churn_probability": round(proba, 4),
            "churn_prediction": proba >= self.threshold,
            "model_version": self.version,
            "model_source": self.source,
        }


def load_model_service() -> ModelService:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(tracking_uri)
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
            logger.info("Loaded %s@%s (v%s) from registry", MODEL_NAME, MODEL_ALIAS, mv.version)
            return ModelService(pipeline, mv.version, "mlflow-registry")
        except Exception as exc:  # fall back rather than fail to start
            logger.warning("Registry load failed (%s); falling back to local model", exc)

    pipeline = joblib.load(LOCAL_MODEL)
    version = "unknown"
    if LOCAL_META.exists():
        version = json.loads(LOCAL_META.read_text()).get("version", "unknown")
    logger.info("Loaded local model v%s from %s", version, LOCAL_MODEL)
    return ModelService(pipeline, version, "local-file")
