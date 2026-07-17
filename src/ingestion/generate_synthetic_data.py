"""
Generates a synthetic customer-churn dataset that mirrors the structure of the
well-known Telco Customer Churn dataset (IBM / Kaggle). Use this for local
development and pipeline testing when you don't yet have the real dataset
wired in.

To use the REAL dataset instead:
  1. Download "Telco Customer Churn" from Kaggle:
     https://www.kaggle.com/datasets/blastchar/telco-customer-churn
  2. Save it as data/raw/telco_churn.csv
  3. Point src/ingestion/load_data.py at that file (see load_data.py).

Run:
    python src/ingestion/generate_synthetic_data.py --n_rows 5000 --out data/raw/telco_churn.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n_rows)  # months, 0-72
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20]
    )
    monthly_charges = np.round(rng.normal(65, 30, n_rows).clip(18, 120), 2)
    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 50, n_rows), 2).clip(0)
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.35, 0.45, 0.20]
    )
    tech_support = rng.choice(["Yes", "No", "No internet service"], n_rows, p=[0.3, 0.5, 0.2])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_rows,
        p=[0.35, 0.2, 0.225, 0.225],
    )
    senior_citizen = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_rows, p=[0.3, 0.7])
    paperless_billing = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])

    # Build churn probability from a realistic latent function, then sample.
    # Month-to-month, fiber, electronic check, short tenure, high charges,
    # no tech support -> higher churn risk (mirrors real-world Telco patterns).
    logit = (
        -1.2
        + 1.4 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.55 * (internet_service == "Fiber optic")
        + 0.5 * (payment_method == "Electronic check")
        + 0.45 * (tech_support == "No")
        - 0.035 * tenure
        + 0.01 * (monthly_charges - 65)
        + 0.25 * (paperless_billing == "Yes")
        - 0.3 * (partner == "Yes")
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_rows) < prob_churn).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n_rows)],
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure": tenure,
            "contract": contract,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "churn": churn,
        }
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/telco_churn.csv")
    args = parser.parse_args()

    df = generate(args.n_rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
