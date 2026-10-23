"""
Unit tests for Module B 168h Drift Prediction (Phase 7).
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from src.drift_prediction import DriftPredictor
from src.feature_engineering import DataLeakageError


@pytest.fixture
def synthetic_drift_df() -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic regression data mimicking early features to 168h target."""
    np.random.seed(42)
    n = 120
    v0 = np.random.normal(10.0, 1.0, n)
    drift = np.random.exponential(0.5, n)
    v24 = v0 + drift
    # 168h is a non-linear accelerated function of 24h drift
    y_168 = v0 + (drift * 7.0) + (drift ** 1.5 * 2.0) + np.random.normal(0, 0.2, n)

    df = pd.DataFrame({
        "Value_0h_uA": v0,
        "Value_24h_uA": v24,
        "Early_Drift_uA": drift,
        "Early_Drift_Rate_uA_per_h": drift / 24.0,
    })
    return df, y_168


def test_drift_predictor_models(synthetic_drift_df):
    """Test fit and evaluation across all supported model types."""
    df, y = synthetic_drift_df
    feat_cols = ["Value_0h_uA", "Value_24h_uA", "Early_Drift_uA", "Early_Drift_Rate_uA_per_h"]

    for model_type in ["linear_regression", "random_forest", "gradient_boosting"]:
        predictor = DriftPredictor(model_type=model_type, feature_names=feat_cols)
        predictor.fit(df, y)
        eval_dict = predictor.evaluate(df, y)

        assert eval_dict["mae_uA"] >= 0.0
        assert eval_dict["rmse_uA"] >= 0.0
        assert eval_dict["r2_score"] > 0.85  # Strong fit on synthetic degradation


def test_drift_predictor_leakage_guard():
    """Verify that using retrospective features raises DataLeakageError."""
    with pytest.raises(DataLeakageError):
        DriftPredictor(feature_names=["Value_0h_uA", "Value_168h_uA"])

    with pytest.raises(DataLeakageError):
        DriftPredictor(feature_names=["Value_0h_uA", "Total_Drift_uA"])


def test_drift_predictor_serialization(synthetic_drift_df, tmp_path):
    """Test model persistence roundtrip."""
    df, y = synthetic_drift_df
    feat_cols = ["Value_0h_uA", "Value_24h_uA"]

    predictor = DriftPredictor(model_type="gradient_boosting", feature_names=feat_cols)
    predictor.fit(df, y)
    preds = predictor.predict(df)

    save_file = tmp_path / "test_drift.pkl"
    predictor.save(save_file)

    loaded = DriftPredictor.load(save_file)
    loaded_preds = loaded.predict(df)

    np.testing.assert_allclose(preds, loaded_preds)
