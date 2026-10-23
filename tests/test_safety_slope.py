"""
Unit tests for Safety Slope and Healthy Degradation Envelope (Phase 8).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.safety_slope import SafetySlopeEstimator


@pytest.fixture
def mock_healthy_df() -> pd.DataFrame:
    """Generate synthetic healthy population data."""
    np.random.seed(42)
    n = 200
    v0 = np.random.normal(10.0, 0.5, n)
    v168 = v0 + np.random.normal(2.0, 0.3, n)  # Drift around 2.0 uA over 168h (~0.012 uA/h)

    return pd.DataFrame({
        "Component_ID": [f"IC_{i:04d}" for i in range(n)],
        "Value_0h_uA": v0,
        "Value_24h_uA": v0 + 0.3,
        "Value_96h_uA": v0 + 1.2,
        "Value_168h_uA": v168,
        "Latent_Defect": [0] * n,
    })


def test_safety_slope_estimator_fit(mock_healthy_df):
    """Test fitting boundaries on healthy components."""
    estimator = SafetySlopeEstimator(method="robust_tukey")
    estimator.fit(mock_healthy_df)

    assert estimator.safety_slope_ > 0.0
    assert "robust_tukey" in estimator.boundary_candidates_
    assert "percentile_99_7" in estimator.boundary_candidates_
    assert "gaussian_3sigma" in estimator.boundary_candidates_
    assert len(estimator.healthy_trajectory_envelope_) == 4


def test_safety_slope_evaluate_components(mock_healthy_df):
    """Test component evaluation and slope breach detection."""
    estimator = SafetySlopeEstimator(method="robust_tukey")
    estimator.fit(mock_healthy_df)

    # Test with one conforming healthy prediction and one defective prediction
    test_df = pd.DataFrame({
        "Component_ID": ["IC_TEST_1", "IC_TEST_2"],
        "Value_0h_uA": [10.0, 10.0],
    })
    pred_168 = np.array([12.0, 30.0])  # IC 1 drift rate = 2/168 = 0.012, IC 2 = 20/168 = 0.119

    eval_df = estimator.evaluate_components(test_df, pred_168)

    assert "Predicted_168h" in eval_df.columns
    assert "Predicted_Drift" in eval_df.columns
    assert "Predicted_Drift_Rate" in eval_df.columns
    assert "Safety_Slope" in eval_df.columns
    assert "Slope_Exceeded" in eval_df.columns

    # IC 1 should be conforming (0), IC 2 should breach slope (1)
    assert eval_df.loc[0, "Slope_Exceeded"] == 0
    assert eval_df.loc[1, "Slope_Exceeded"] == 1


def test_safety_slope_serialization(mock_healthy_df, tmp_path):
    """Test persistence roundtrip."""
    estimator = SafetySlopeEstimator(method="percentile_99_7")
    estimator.fit(mock_healthy_df)

    save_path = tmp_path / "safety_slope.pkl"
    estimator.save(save_path)

    loaded = SafetySlopeEstimator.load(save_path)
    assert loaded.method == "percentile_99_7"
    assert loaded.safety_slope_ == estimator.safety_slope_
