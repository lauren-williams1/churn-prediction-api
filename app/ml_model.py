"""
ML MODEL INFERENCE LOGIC
========================

Purpose:
    Encapsulates all ML prediction logic separate from API code.
    This separation follows clean architecture principles:
    - API layer (main.py) handles HTTP
    - ML layer (ml_model.py) handles predictions
    
Benefits:
    - Easier testing (test ML logic without HTTP)
    - Easier model updates (swap models without changing API)
    - Cleaner code organization
    - Reusable in non-API contexts (batch jobs, notebooks)

Functions:
    - preprocess_input(): Transform raw input to model format
    - make_prediction(): Run model inference
    - explain_prediction(): Generate human-readable explanation

Connection to Other Files:
    ← Loads: models/*.pkl (created by train_model.py)
    ← Used by: app/main.py (imports these functions)
    → Returns: Predictions and explanations
    
Example Usage in main.py:
    from app.ml_model import make_prediction
    
    @app.post("/predict")
    def predict(customer: CustomerFeatures):
        result = make_prediction(customer.dict())
        return result
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any

def preprocess_input(customer_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Transform raw customer data into model-ready format.
    
    Steps:
        1. Create DataFrame from dict
        2. Encode categorical variables using saved encoders
        3. Ensure features are in correct order
        4. Handle any missing values
    
    Args:
        customer_data: Dict with customer features
        
    Returns:
        DataFrame ready for model.predict()
        
    Example:
        >>> data = {"gender": "Male", "tenure": 12, ...}
        >>> X = preprocess_input(data)
        >>> X.shape
        (1, 19)  # 1 row, 19 features
    """
    # TO BE IMPLEMENTED TOMORROW
    pass

def make_prediction(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make churn prediction for a single customer.
    
    Process:
        1. Preprocess input
        2. Get model prediction
        3. Get prediction probability
        4. Determine risk level
        5. Generate recommendation
    
    Args:
        customer_data: Dict with customer features
        
    Returns:
        Dict with prediction results:
            - will_churn: bool
            - churn_probability: float
            - confidence: float
            - risk_level: str
            - recommendation: str
    
    Example:
        >>> result = make_prediction(customer_dict)
        >>> result['will_churn']
        True
        >>> result['churn_probability']
        0.78
    """
    # TO BE IMPLEMENTED TOMORROW
    pass

def explain_prediction(probability: float) -> tuple:
    """
    Generate risk level and recommendation based on probability.
    
    Business Logic:
        - probability < 0.3: LOW risk
        - 0.3 <= probability < 0.7: MEDIUM risk
        - probability >= 0.7: HIGH risk
    
    Args:
        probability: Churn probability from model (0-1)
        
    Returns:
        tuple: (risk_level, recommendation)
        
    Example:
        >>> explain_prediction(0.78)
        ('HIGH', 'Immediate retention campaign recommended')
    """
    # TO BE IMPLEMENTED TOMORROW
    pass