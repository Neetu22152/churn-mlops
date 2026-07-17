"""
Basic test suite. Run with: pytest tests/ -v

These are intentionally lightweight (fast, no external services) so they run
cleanly in GitHub Actions CI on every push.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.generate_synthetic_data import generate
from src.features.build_features import engineer_features, build_training_frame, FeatureEncoder


@pytest.fixture
def sample_df():
    return generate(n_rows=200, seed=1)


def test_generate_synthetic_data_shape(sample_df):
    assert len(sample_df) == 200
    assert "churn" in sample_df.columns
    assert sample_df["churn"].isin([0, 1]).all()


def test_engineer_features_adds_expected_columns(sample_df):
    df = engineer_features(sample_df)
    for col in ["avg_monthly_spend", "tenure_bucket", "is_month_to_month", "high_value_customer"]:
        assert col in df.columns


def test_engineer_features_no_divide_by_zero(sample_df):
    df = sample_df.copy()
    df.loc[0, "tenure"] = 0
    result = engineer_features(df)
    assert not result["avg_monthly_spend"].isna().any()
    assert not (result["avg_monthly_spend"] == float("inf")).any()


def test_build_training_frame_shapes(sample_df):
    X, y, encoder = build_training_frame(sample_df)
    assert len(X) == len(y) == len(sample_df)
    assert isinstance(encoder, FeatureEncoder)


def test_encoder_handles_unseen_category_gracefully(sample_df):
    """Simulates train/serve skew scenario: a category value at inference
    time that wasn't in the training set shouldn't crash the pipeline."""
    X_train, y_train, encoder = build_training_frame(sample_df)

    new_row = sample_df.iloc[[0]].copy()
    new_row["payment_method"] = "Crypto"  # unseen at training time
    new_row_fe = engineer_features(new_row)
    X_new = encoder.transform(new_row_fe)

    # Should have the same columns as training, no crash, no NaNs introduced.
    assert list(X_new.columns) == encoder.fitted_columns_
    assert not X_new.isna().any().any()


def test_encoder_transform_before_fit_raises():
    encoder = FeatureEncoder()
    with pytest.raises(RuntimeError):
        encoder.transform(pd.DataFrame())
