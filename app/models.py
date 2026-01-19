"""
PYDANTIC DATA MODELS
====================

Purpose:
    Define request/response schemas for API endpoints.
    Pydantic provides automatic validation, serialization,
    and documentation generation.

Why Pydantic?
    - Type validation: Ensures inputs are correct type
    - Auto docs: FastAPI uses these for /docs page
    - IDE support: Get autocomplete for request fields
    - Error messages: Clear validation errors for users

Models Defined:
    1. CustomerFeatures: Input schema for /predict endpoint
    2. PredictionResponse: Output schema for predictions
    3. BatchPredictionRequest: Input for batch predictions (future)

Connection to Other Files:
    ← Used by: app/main.py (import these models)
    → Validates: User input from HTTP requests
    
Example Usage in main.py:
    from app.models import CustomerFeatures, PredictionResponse
    
    @app.post("/predict", response_model=PredictionResponse)
    def predict(customer: CustomerFeatures):
        # customer is validated and typed
        pass
"""

from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

class CustomerFeatures(BaseModel):
    """
    Input schema for single customer prediction.
    
    All fields match the training data columns.
    Types and constraints ensure data quality.
    
    Attributes:
        gender: Customer gender
        SeniorCitizen: 0 or 1 (boolean as int)
        Partner: Whether customer has a partner
        Dependents: Whether customer has dependents
        tenure: Months as customer (0-72)
        PhoneService: Whether has phone service
        MultipleLines: Multiple phone lines status
        InternetService: Type of internet service
        OnlineSecurity: Online security addon
        OnlineBackup: Online backup addon
        DeviceProtection: Device protection addon
        TechSupport: Tech support addon
        StreamingTV: Streaming TV addon
        StreamingMovies: Streaming movies addon
        Contract: Contract type
        PaperlessBilling: Paperless billing status
        PaymentMethod: Payment method
        MonthlyCharges: Monthly charge amount
        TotalCharges: Total charges to date
    
    Example:
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "tenure": 12,
            "MonthlyCharges": 65.50,
            ...
        }
    """
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=72, description="Months as customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(gt=0, description="Monthly charge in dollars")
    TotalCharges: float = Field(ge=0, description="Total charges to date")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 844.20
            }
        }

class PredictionResponse(BaseModel):
    """
    Output schema for prediction endpoint.
    
    Returns prediction, probability, and explanation.
    
    Attributes:
        will_churn: Boolean prediction (True = will churn)
        churn_probability: Probability of churn (0-1)
        confidence: Model confidence (0-1)
        risk_level: Human-readable risk level
        recommendation: Suggested action
    
    Example:
        {
            "will_churn": true,
            "churn_probability": 0.78,
            "confidence": 0.78,
            "risk_level": "HIGH",
            "recommendation": "Immediate retention campaign recommended"
        }
    """
    will_churn: bool
    churn_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommendation: str
    
    class Config:
        schema_extra = {
            "example": {
                "will_churn": True,
                "churn_probability": 0.78,
                "confidence": 0.78,
                "risk_level": "HIGH",
                "recommendation": "Immediate retention campaign recommended"
            }
        }