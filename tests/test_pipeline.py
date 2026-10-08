"""Run with:  python -m pytest -q   (from the project root)"""
import numpy as np
import pandas as pd
import pytest as pytest

from app.model_service import ModelService
from promote_model import should_promote
from train_model import NUMERIC, build_candidates


def _toy_data(n=400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "SeniorCitizen": rng.integers(0, 2, n),
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n),
            "TotalCharges": rng.uniform(0, 8000, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        }
    )
    y = ((X["Contract"] == "Month-to-month") & (X["tenure"] < 24)).astype(int)
    return X, y


CATEGORICAL = ["Contract", "mlflow ui --backend-store-uri sqlite:///mlflow.dbInternetService"]


@pytest.mark.parametrize("name", ["random_forest", "logistic_regression"])
def test_pipeline_fits_and_handles_unseen_category(name):
    X, y = _toy_data()
    pipe = build_candidates(CATEGORICAL)[name].fit(X, y)
    row = X.iloc[[0]].copy()
    row["Contract"] = "Brand New Contract Type"  # unseen at training time
    proba = pipe.predict_proba(row)[0, 1]
    assert 0.0 <= proba <= 1.0


def test_pipeline_learns_signal():
    from sklearn.metrics import roc_auc_score

    X, y = _toy_data()
    pipe = build_candidates(CATEGORICAL)["random_forest"].fit(X, y)
    assert roc_auc_score(y, pipe.predict_proba(X)[:, 1]) > 0.9


def test_gate_promotes_when_no_champion_and_floors_met():
    ok, reasons = should_promote({"test_roc_auc": 0.84, "test_recall": 0.74}, None)
    assert ok and not reasons


def test_gate_rejects_regression_vs_champion():
    cand = {"test_roc_auc": 0.83, "test_recall": 0.74}
    champ = {"test_roc_auc": 0.85, "test_recall": 0.74}
    ok, reasons = should_promote(cand, champ)
    assert not ok and any("regresses" in r for r in reasons)


def test_gate_rejects_below_absolute_floor():
    ok, reasons = should_promote({"test_roc_auc": 0.70, "test_recall": 0.74}, None)
    assert not ok


def test_gate_rejects_low_recall():
    ok, reasons = should_promote({"test_roc_auc": 0.85, "test_recall": 0.40}, None)
    assert not ok and any("recall" in r for r in reasons)


def test_model_service_output_shape():
    X, y = _toy_data()
    pipe = build_candidates(CATEGORICAL)["logistic_regression"].fit(X, y)
    svc = ModelService(pipe, version="1", source="test")
    out = svc.predict(X.iloc[0].to_dict())
    assert set(out) == {
        "churn_probability",
        "churn_prediction",
        "model_version",
        "model_source",
    }
    assert 0.0 <= out["churn_probability"] <= 1.0
