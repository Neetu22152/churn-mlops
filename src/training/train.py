"""
Trains the churn model with MLflow experiment tracking: params, metrics, the
model artifact, and the fitted FeatureEncoder are all logged together so any
run can be reproduced or promoted to production.

Requires: pip install mlflow scikit-learn xgboost pandas
Run:
    python -m src.training.train --model xgboost
    python -m src.training.train --model logreg   # fast baseline
    mlflow ui   # then open http://localhost:5000 to browse runs
"""
import argparse
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)

from src.ingestion.load_data import load_raw_data
from src.features.build_features import build_training_frame

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_model(name: str):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                eval_metric="logloss", random_state=42,
            )
        except ImportError:
            print("xgboost not installed — falling back to GradientBoostingClassifier.")
            return GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train(model_name: str = "xgboost", test_size: float = 0.2, seed: int = 42):
    df = load_raw_data()
    X, y, encoder = build_training_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = get_model(model_name)

    run_ctx = mlflow.start_run(run_name=model_name) if MLFLOW_AVAILABLE else _NullContext()
    with run_ctx:
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        print(f"[{model_name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if MLFLOW_AVAILABLE:
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(
                model, "model",serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
            # Log the feature encoder too — it must travel with the model so
            # serving code can reproduce the exact same feature vector.
            with tempfile.TemporaryDirectory() as tmp:
                enc_path = Path(tmp) / "encoder.pkl"
                with open(enc_path, "wb") as f:
                    pickle.dump(encoder, f)
                mlflow.log_artifact(str(enc_path))

    # Always also save locally to models/ so serving works even without
    # a running MLflow tracking server (e.g. simple Docker deployment).
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(MODEL_DIR / "training_columns.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    print(f"Saved model + encoder to {MODEL_DIR}/")
    return model, encoder, metrics


class _NullContext:
    """No-op context manager used when mlflow isn't installed, so the script
    still runs (with a printed warning) rather than hard-failing."""
    def __enter__(self):
        print("NOTE: mlflow not installed — training will run without experiment tracking.")
        return self
    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xgboost", choices=["logreg", "random_forest", "xgboost"])
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("churn-prediction")

    train(model_name=args.model, test_size=args.test_size)
