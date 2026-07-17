"""
Feature engineering. Kept as a pure, deterministic function of a dataframe so
it can be called identically at training time and at inference time — this
is what prevents training/serving skew, one of the most common real-world
MLOps bugs.
"""
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

CATEGORICAL_COLS = [
    "partner", "dependents", "contract", "paperless_billing",
    "payment_method", "internet_service", "tech_support",
]
NUMERIC_COLS = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]
ENGINEERED_COLS = ["avg_monthly_spend", "tenure_bucket", "is_month_to_month", "high_value_customer"]

TARGET_COL = "churn"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Called on both train and inference data."""
    df = df.copy()

    # Avoid divide-by-zero for brand-new customers (tenure == 0).
    df["avg_monthly_spend"] = df["total_charges"] / df["tenure"].replace(0, 1)

    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "13-24mo", "25-48mo", "49-72mo"],
    ).astype(str)

    df["is_month_to_month"] = (df["contract"] == "Month-to-month").astype(int)

    # "High value" = top-quartile monthly spend — a flag that's useful both
    # as a feature and later for business-facing dashboards (e.g. "we're
    # about to lose high-value customers").
    threshold = df["monthly_charges"].quantile(0.75)
    df["high_value_customer"] = (df["monthly_charges"] >= threshold).astype(int)

    return df


@dataclass
class FeatureEncoder:
    """One-hot encodes categoricals, remembering columns seen at fit time so
    inference always produces the same feature vector shape, even if a rare
    category is missing from a single prediction request.
    """
    categorical_cols: list = field(default_factory=lambda: CATEGORICAL_COLS + ["tenure_bucket"])
    numeric_cols: list = field(default_factory=lambda: NUMERIC_COLS + [
        "avg_monthly_spend", "is_month_to_month", "high_value_customer"
    ])
    fitted_columns_: list = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(df[self.categorical_cols], drop_first=False)
        X = pd.concat([df[self.numeric_cols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        self.fitted_columns_ = X.columns.tolist()
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_columns_ is None:
            raise RuntimeError("Call fit_transform on training data before transform().")
        X = pd.get_dummies(df[self.categorical_cols], drop_first=False)
        X = pd.concat([df[self.numeric_cols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        # Align to training-time columns: add missing (as 0), drop unseen extras.
        X = X.reindex(columns=self.fitted_columns_, fill_value=0)
        return X


def build_training_frame(df: pd.DataFrame):
    """Convenience: engineer features + encode + split target, in one call."""
    df = engineer_features(df)
    encoder = FeatureEncoder()
    X = encoder.fit_transform(df)
    y = df[TARGET_COL]
    return X, y, encoder


if __name__ == "__main__":
    from src.ingestion.load_data import load_raw_data

    df = load_raw_data()
    X, y, encoder = build_training_frame(df)
    print("Feature matrix shape:", X.shape)
    print("Columns:", X.columns.tolist())
