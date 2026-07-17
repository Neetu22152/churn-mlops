from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telco_churn.csv"

EXPECTED_COLUMNS = [
    "customer_id", "senior_citizen", "partner", "dependents", "tenure",
    "contract", "paperless_billing", "payment_method", "internet_service",
    "tech_support", "monthly_charges", "total_charges", "churn",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    
    if not path.exists():
        raise FileNotFoundError(
            f"No data found at {path}. Run "
            "`python src/ingestion/generate_synthetic_data.py` first, "
            "or place the real dataset there."
        )

    df = pd.read_csv(path)

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing expected columns: {missing}. "
            "NOTE: if you downloaded the raw Kaggle Telco CSV, its columns "
            "are CamelCase (e.g. 'customerID', 'MonthlyCharges'). Run "
            "`normalize_kaggle_columns()` below on it first."
        )

    # Basic sanity cleaning that's needed even on the real dataset:
    # TotalCharges in the real Kaggle CSV has blank strings for tenure==0.
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    df["total_charges"] = df["total_charges"].fillna(0)

    return df


def normalize_kaggle_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "customerID": "customer_id",
        "SeniorCitizen": "senior_citizen",
        "Partner": "partner",
        "Dependents": "dependents",
        "tenure": "tenure",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "InternetService": "internet_service",
        "TechSupport": "tech_support",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Churn": "churn",
    }
    df = df.rename(columns=rename_map)
    if df["churn"].dtype == object:
        df["churn"] = (df["churn"] == "Yes").astype(int)
    keep = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df[keep]


if __name__ == "__main__":
    df = load_raw_data()
    print(df.shape)
    print(df.head())
    print(df["churn"].value_counts(normalize=True))
