# Customer Churn Prediction — End-to-End MLOps Pipeline

A production-style MLOps pipeline for predicting telecom customer churn —
built to demonstrate the full lifecycle of a machine learning system, not
just a trained model in a notebook.

## Why this project

Most ML portfolios stop at "I trained a model and got 90% accuracy." This
one goes further and answers the questions a hiring team actually cares
about: How does the model get deployed? How do you know when it's wrong?
How does it get retrained? What breaks if the input data changes shape?

## Architecture

```
Raw data (CSV/DB)
      │
      ▼
Ingestion (src/ingestion) ── schema validation
      │
      ▼
Feature engineering (src/features) ── shared by training AND serving
      │
      ▼
Training (src/training) ── MLflow experiment tracking
      │
      ├──► models/ (versioned artifacts: model, encoder, schema)
      │
      ▼
Serving (src/serving) ── FastAPI, Dockerized
      │
      ▼
Monitoring (src/monitoring) ── Evidently drift detection
      │
      └──► triggers retraining when drift is detected
```

**Key design decision:** feature engineering (`src/features/build_features.py`)
is a pure function called identically at training time and inference time.
This eliminates train/serve skew — a common real-world bug where the
production API computes features slightly differently than training did.

## Stack

| Layer | Tool |
|---|---|
| Data | pandas, synthetic Telco-style generator (swap in real Kaggle dataset) |
| Experiment tracking | MLflow |
| Models | scikit-learn, XGBoost |
| Serving | FastAPI + Docker |
| Monitoring | Evidently AI (data drift detection) |
| CI/CD | GitHub Actions |

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (or drop the real Kaggle Telco CSV into data/raw/telco_churn.csv —
#    see src/ingestion/load_data.py for the column-normalization helper)
python src/ingestion/generate_synthetic_data.py --n_rows 5000

# 3. Train (with MLflow tracking)
python -m src.training.train --model xgboost
mlflow ui   # browse experiments at http://localhost:5000

# 4. Serve
uvicorn src.serving.app:app --reload --port 8000
# then POST to http://localhost:8000/predict — see /docs for the schema

# 5. Check for drift (simulate "new" data first)
python -m src.monitoring.check_drift --current data/raw/telco_churn_recent.csv
```

## Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Testing

```bash
pytest tests/ -v
```

## Using the real dataset

This repo ships with a synthetic data generator so the pipeline runs out of
the box. To use the real, well-known **Telco Customer Churn** dataset:

1. Download from Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Load it and run `normalize_kaggle_columns()` from `src/ingestion/load_data.py`
   to map its CamelCase columns to this project's schema.
3. Save the result to `data/raw/telco_churn.csv`.

Everything downstream (features, training, serving, monitoring) works
unchanged.

## What this project demonstrates

- End-to-end pipeline design, not just modeling
- Train/serve consistency (shared feature engineering code)
- Experiment tracking and reproducibility (MLflow)
- Production API design (FastAPI, input validation, health checks)
- Containerization (Docker, docker-compose)
- Automated testing and CI/CD (GitHub Actions)
- Data drift monitoring — the piece that closes the loop and justifies
  calling this "MLOps" rather than just "ML"

## Possible extensions

- Airflow DAG to schedule retraining when drift is detected
- Feature store (Feast) if scaling to multiple models
- A/B testing framework to evaluate retention interventions on high-risk
  customers (ties in causal inference)
- Model explainability (SHAP) surfaced in the API response
