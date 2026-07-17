"""
FastAPI serving layer for the churn model.

Run locally:
    uvicorn src.serving.app:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API docs.

Design notes:
- Loads model + encoder once at startup (not per-request) for latency.
- Uses the SAME engineer_features()/encoder.transform() path as training,
  eliminating train/serve skew.
- /health is separate from /predict so container orchestrators (Docker,
  k8s) can health-check without exercising the model.
"""
import pickle
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import engineer_features

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts customer churn risk for a telecom subscriber.",
    version="1.0.0",
)

_model = None
_encoder = None


class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., example="CUST-000123")
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: Literal["Yes", "No"]
    payment_method: Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    tech_support: Literal["Yes", "No", "No internet service"]
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_tier: str


def _risk_tier(prob: float) -> str:
    if prob >= 0.7:
        return "high"
    if prob >= 0.4:
        return "medium"
    return "low"


@app.on_event("startup")
def load_artifacts():
    global _model, _encoder
    try:
        with open(MODEL_DIR / "model.pkl", "rb") as f:
            _model = pickle.load(f)
        with open(MODEL_DIR / "encoder.pkl", "rb") as f:
            _encoder = pickle.load(f)
    except FileNotFoundError:
        # Don't crash the app — /health will report not-ready, and /predict
        # will return a clear 503 instead of a confusing stack trace.
        _model, _encoder = None, None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures):
    if _model is None or _encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first: python -m src.training.train",
        )

    df = pd.DataFrame([customer.dict()])
    df = engineer_features(df)
    X = _encoder.transform(df)

    proba = float(_model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    return ChurnPrediction(
        customer_id=customer.customer_id,
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        risk_tier=_risk_tier(proba),
    )
