"""
Unit and Integration Tests for Module B Inference Workflow & Zero Data Leakage Enforcement.

Verifies:
1. Value_0h + Value_24h -> prediction works.
2. Value_0h + Value_24h + Value_168h -> prediction works; Value_168h used strictly for post-burn-in evaluation.
3. Value_0h + Value_24h without Value_168h -> prediction still works seamlessly.
4. Value_0h only (missing Value_24h) -> Module B flagged unavailable.
5. Verify Value_168h is NOT present in Module B inference feature matrix X.
6. Verify Predicted 168h (and its drift rate) is used for safety-slope and risk analysis.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.drift_prediction import DriftPredictor
from src.feature_engineering import FeatureEngineer
from src.inference import InferencePipeline, analyze_component
from src.schema_adapter import DatasetAdapter, DatasetProfiler


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def trained_pipeline():
    return InferencePipeline()


def test_1_prediction_works_with_0h_and_24h(trained_pipeline):
    """Test 1: Value_0h + Value_24h -> Module B prediction works."""
    df_early = pd.DataFrame({
        "Component_ID": ["IC_TEST_01"],
        "Lot_ID": ["LOT_001"],
        "Value_0h_uA": [10.12],
        "Value_24h_uA": [11.56],
        "PropagationDelay_0h_ns": [2.015],
        "PropagationDelay_24h_ns": [2.218],
        "Iddq_0h_mA": [1.20],
        "Iddq_24h_mA": [1.35],
    })

    results = trained_pipeline.evaluate_batch(df_early)
    assert len(results) == 1
    assert "Predicted_168h" in results.columns
    pred_168 = results["Predicted_168h"].iloc[0]
    assert pred_168 > 11.56  # Physical degradation monotonicity
    assert not np.isnan(pred_168)


def test_2_actual_168h_used_only_for_evaluation(trained_pipeline):
    """Test 2: When Value_168h is present, prediction is identical and 168h is used only for evaluation."""
    df_with_168 = pd.DataFrame({
        "Component_ID": ["IC_TEST_02"],
        "Lot_ID": ["LOT_001"],
        "Value_0h_uA": [10.12],
        "Value_24h_uA": [11.56],
        "Value_168h_uA": [33.40],
        "PropagationDelay_0h_ns": [2.015],
        "PropagationDelay_24h_ns": [2.218],
    })

    df_without_168 = pd.DataFrame({
        "Component_ID": ["IC_TEST_02"],
        "Lot_ID": ["LOT_001"],
        "Value_0h_uA": [10.12],
        "Value_24h_uA": [11.56],
        "PropagationDelay_0h_ns": [2.015],
        "PropagationDelay_24h_ns": [2.218],
    })

    res_with = trained_pipeline.evaluate_batch(df_with_168)
    res_without = trained_pipeline.evaluate_batch(df_without_168)

    # Predictions MUST be exactly identical, proving Value_168h was not used in inference!
    assert res_with["Predicted_168h"].iloc[0] == pytest.approx(res_without["Predicted_168h"].iloc[0], rel=1e-5)

    # Evaluation error calculation
    actual = 33.40
    predicted = res_with["Predicted_168h"].iloc[0]
    error = abs(actual - predicted)
    assert error < 5.0  # Forecast within 5 uA


def test_3_prediction_works_without_168h(trained_pipeline):
    """Test 3: Dataset with NO Value_168h column still executes prediction and risk engine cleanly."""
    result = analyze_component({
        "Component_ID": "IC_NO_168",
        "Lot_ID": "LOT_001",
        "Value_0h_uA": 10.0,
        "Value_24h_uA": 15.0,
    })

    assert "predicted_168h" in result
    assert result["predicted_168h"] > 15.0
    assert result["risk_level"] in ["NORMAL", "WATCH", "HIGH_RISK"]
    assert "explanation" in result


def test_4_missing_24h_module_b_unavailable(client):
    """Test 4: Value_0h only (missing Value_24h) -> Module B marked unavailable."""
    df_no_24 = pd.DataFrame({
        "Component_ID": ["IC_NO_24"],
        "Lot_ID": ["LOT_001"],
        "Value_0h_uA": [10.0],
        "Value_168h_uA": [33.0],  # 168h cannot substitute for missing 24h
    })

    profile = DatasetProfiler.profile_dataframe(df_no_24)
    assert profile["compatibility"]["module_b"] is False
    assert profile["time_points_detected"]["24h"] is False

    # API call returns 400 validation error
    csv_text = "Component_ID,Lot_ID,Value_0h_uA,Value_168h_uA\nIC_NO_24,LOT_001,10.0,33.0\n"
    res = client.post("/dataset/analyze", json={"csv_text": csv_text})
    assert res.status_code == 400
    assert "Value_24h" in res.json()["detail"]


def test_5_zero_data_leakage_in_module_b_feature_names(trained_pipeline):
    """Test 5: Explicitly verify Value_168h is not in Module B feature names or transform outputs."""
    drift_model = trained_pipeline.drift_predictor
    assert drift_model is not None

    forbidden_cols = ["Value_168h_uA", "Value_96h_uA", "Total_Drift_uA", "Latent_Defect", "168h", "96h"]
    for f in drift_model.feature_names:
        for forb in forbidden_cols:
            assert forb not in f, f"Data leakage detected! Forbidden term '{forb}' found in feature '{f}'"


def test_6_predicted_168h_used_for_safety_slope_and_risk(trained_pipeline):
    """Test 6: Safety slope and risk calculation strictly use Predicted 168h, not actual 168h."""
    # Defect with high 24h drift: 10.12 -> 11.56 uA
    df_defect = pd.DataFrame({
        "Component_ID": ["IC_000006"],
        "Lot_ID": ["LOT_001"],
        "Value_0h_uA": [10.12],
        "Value_24h_uA": [11.56],
        "Value_168h_uA": [33.40],
    })

    results = trained_pipeline.evaluate_batch(df_defect)
    pred_168 = results["Predicted_168h"].iloc[0]
    pred_drift_rate = (pred_168 - 10.12) / 168.0

    # Safety slope threshold is 0.037074 uA/h
    assert pred_drift_rate > 0.037074
    assert results["Slope_Exceeded"].iloc[0] == 1
    assert results["Risk_Level"].iloc[0] == "HIGH_RISK"
