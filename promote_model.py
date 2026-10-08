"""Promotion gate: move the 'candidate' alias to 'champion' only if it passes checks.

Exit code 0 = promoted (or already champion), 1 = rejected. CI uses the exit code.

Gate rules:
  - candidate test ROC-AUC >= MIN_AUC (absolute floor)
  - candidate test recall  >= MIN_RECALL (we care about catching churners)
  - candidate test ROC-AUC >= champion test ROC-AUC + MIN_DELTA (no regression)
Note: champion and candidate are compared on the same fixed test split
(stratified, random_state=42), so the comparison is like-for-like.
"""
import os
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "churn-model")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MIN_AUC = float(os.getenv("MIN_AUC", "0.80"))
MIN_RECALL = float(os.getenv("MIN_RECALL", "0.65"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "0.0"))


def should_promote(
    candidate: dict,
    champion: dict, 
    min_auc: float = MIN_AUC,
    min_recall: float = MIN_RECALL,
    min_delta: float = MIN_DELTA,
) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["test_roc_auc"] < min_auc:
        reasons.append(f"AUC {candidate['test_roc_auc']:.4f} below floor {min_auc}")
    if candidate["test_recall"] < min_recall:
        reasons.append(f"recall {candidate['test_recall']:.4f} below floor {min_recall}")
    if champion is not None:
        needed = champion["test_roc_auc"] + min_delta
        if candidate["test_roc_auc"] < needed:
            reasons.append(
                f"AUC {candidate['test_roc_auc']:.4f} regresses vs champion "
                f"{champion['test_roc_auc']:.4f} (need >= {needed:.4f})"
            )
    return (len(reasons) == 0, reasons)


def _get_alias(client: MlflowClient, alias: str):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias)
    except MlflowException:
        return None


def main() -> int:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    cand = _get_alias(client, "candidate")
    if cand is None:
        print("No 'candidate' alias found. Run train_model.py first.")
        return 1
    champ = _get_alias(client, "champion")

    if champ is not None and champ.version == cand.version:
        print(f"v{cand.version} is already champion.")
        return 0

    cand_metrics = client.get_run(cand.run_id).data.metrics
    champ_metrics = client.get_run(champ.run_id).data.metrics if champ else None

    ok, reasons = should_promote(cand_metrics, champ_metrics)
    if ok:
        client.set_registered_model_alias(MODEL_NAME, "champion", cand.version)
        prev = f"v{champ.version}" if champ else "none"
        print(f"PROMOTED v{cand.version} to champion (previous: {prev})")
        return 0

    print(f"REJECTED v{cand.version}:")
    for r in reasons:
        print(f"  - {r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
