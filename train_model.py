"""Churn training pipeline: CV model selection, MLflow tracking, model registry.

Run:  python train_model.py
Then: python promote_model.py   (gate: candidate must match or beat champion)
"""

"""
MODEL TRAINING PIPELINE
========================

Purpose:
    Complete end-to-end pipeline for training a churn prediction model.
    Handles data preprocessing, feature engineering, model training,
    evaluation, and artifact saving.

Input:
    - data/telco_churn.csv (raw customer data)

Processing Steps:
    1. Load data from CSV
    2. Drop customerID (not a feature)
    3. Handle TotalCharges data type issues
    4. Encode categorical variables (LabelEncoder)
    5. Split into train/test (80/20 stratified)
    6. Train RandomForestClassifier with class balancing
    7. Evaluate on test set
    8. Save model artifacts

Output Files (saved to models/):
    - model.pkl: Trained RandomForestClassifier
    - label_encoders.pkl: Dict of LabelEncoders for each categorical feature
    - feature_names.pkl: List of feature names in correct order
    - metadata.pkl: Dict with accuracy, AUC, feature list, model type

Model Hyperparameters:
    - n_estimators: 100 trees
    - max_depth: 10 (prevents overfitting)
    - class_weight: 'balanced' (handles 27% churn imbalance)
    - random_state: 42 (reproducibility)

Expected Performance:
    - Accuracy: ~80%
    - ROC-AUC: ~0.84
    - Recall (Churn): ~0.55 (catches 55% of churners)
    - Precision (Churn): ~0.65 (65% of churn predictions are correct)

Connection to Other Files:
    ← Reads: data/telco_churn.csv (from download_data.py)
    → Creates: models/*.pkl (used by app/main.py)
    
Why RandomForest?
    - Handles non-linear relationships
    - Robust to outliers
    - No feature scaling needed
    - Good accuracy vs speed trade-off
    - Built-in feature importance

When to Run:
    - Initially to create model
    - When retraining with new data
    - When tuning hyperparameters

Example Usage:
    $ python3 train_model.py
    🚀 Training Churn Prediction Model...
    ✅ Loaded 7043 customers
    ✅ Preprocessed 19 features
    🤖 Training Random Forest...
    📊 Results:
       Accuracy: 0.805
       ROC-AUC: 0.847
    ✅ Model saved to models/
"""

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_PATH = Path(os.getenv("DATA_PATH", "data/telco_churn.csv"))
MODELS_DIR = Path("models")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "churn-prediction"
RANDOM_STATE = 42
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"], errors="ignore")
    # Telco has blank TotalCharges for tenure-0 customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    categorical = [c for c in X.columns if c not in NUMERIC]
    return X, y, categorical


def build_candidates(categorical):
    """Each candidate is one Pipeline: preprocessing + model, saved as one artifact."""
    rf_pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            ),
        ]
    )
    lr_pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return {
        "random_forest": Pipeline(
            [
                ("pre", rf_pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("pre", lr_pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X, y, categorical = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Model selection on TRAIN ONLY via stratified CV; test set is touched once below.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = build_candidates(categorical)
    cv_auc = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        cv_auc[name] = (float(scores.mean()), float(scores.std()))
        print(f"{name}: CV AUC {scores.mean():.4f} +/- {scores.std():.4f}")
    best = max(cv_auc, key=lambda n: cv_auc[n][0])
    pipe = candidates[best]

    with mlflow.start_run(run_name=f"train-{best}") as run:
        mlflow.log_params(
            {
                "model_type": best,
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "churn_rate": round(float(y.mean()), 4),
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "threshold": 0.5,
            }
        )
        for name, (mean, std) in cv_auc.items():
            mlflow.log_metric(f"cv_roc_auc_{name}", mean)
            mlflow.log_metric(f"cv_roc_auc_std_{name}", std)

        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "test_roc_auc": roc_auc_score(y_te, proba),
            "test_accuracy": accuracy_score(y_te, pred),
            "test_precision": precision_score(y_te, pred),
            "test_recall": recall_score(y_te, pred),
            "test_f1": f1_score(y_te, pred),
        }
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        mlflow.log_dict(
            {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "confusion_matrix.json",
        )

        mlflow.sklearn.log_model(
            pipe, artifact_path="model", registered_model_name=MODEL_NAME
        )

        client = MlflowClient()
        versions = client.search_model_versions(f"run_id='{run.info.run_id}'")
        version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(MODEL_NAME, "candidate", str(version))

        # Local export so the Docker image can run without an MLflow server
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipe, MODELS_DIR / "pipeline.joblib")
        (MODELS_DIR / "metadata.json").write_text(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "model_type": best,
                    "version": str(version),
                    "run_id": run.info.run_id,
                    "metrics": {k: float(v) for k, v in metrics.items()},
                },
                indent=2,
            )
        )

    print(f"\nRegistered {MODEL_NAME} v{version} as 'candidate' ({best})")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("Next: python promote_model.py")


if __name__ == "__main__":
    main()
