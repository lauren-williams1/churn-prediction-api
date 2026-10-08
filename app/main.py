"""Churn prediction API. Serves the champion model via ModelService."""
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.model_service import load_model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    model_version: str
    model_source: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model_service()
    logger.info(
        "Model ready: version=%s source=%s",
        app.state.model.version,
        app.state.model.source,
    )
    yield


app = FastAPI(title="Churn Prediction API", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_version": model.version,
        "model_source": model.source,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return model.predict(features.model_dump())
