"""
Unit tests for End-to-End ML Inference Pipeline (Phase 11).

Validates the 6 required screening scenarios:
1. Normal component
2. Strong anomaly
3. Latent defect below datasheet limit
4. Component exceeding datasheet limit
5. Missing measurement
6. Extreme / corrupted measurement
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.inference import InferencePipeline, analyze_component


@pytest.fixture(scope="module")
def pipeline() -> InferencePipeline:
    """Initialize shared inference pipeline instance."""
    return InferencePipeline()


# ==============================================================================
# Scenario 1: Normal Component
# ==============================================================================
def test_scenario_1_normal_component(pipeline):
    """Scenario 1: Conforming healthy component with minimal thermal drift."""
    normal_data = {
        "Component_ID": "IC_HEALTHY_TEST_01",
        "Lot_ID": "LOT_001",
        "Value_0h_uA": 10.10,
        "Value_24h_uA": 10.35,
        "PropagationDelay_0h_ns": 2.012,
        "PropagationDelay_24h_ns": 2.025,
        "Iddq_0h_mA": 1.20,
        "Iddq_24h_mA": 1.21,
        "Datasheet_Leakage_Limit_uA": 50.0,
    }

    res = pipeline.analyze_component(normal_data)

    assert res["component_id"] == "IC_HEALTHY_TEST_01"
    assert res["risk_level"] == "NORMAL"
    assert res["risk_score"] < 35.0
    assert res["datasheet_pass"] is True
    assert res["slope_exceeded"] == 0
    assert len(res["explanation"]["human_readable_reasons"]) > 0


# ==============================================================================
# Scenario 2: Strong Dynamic Anomaly
# ==============================================================================
def test_scenario_2_strong_anomaly(pipeline):
    """Scenario 2: Component exhibiting rapid kinetic acceleration and high anomaly score."""
    anomaly_data = {
        "Component_ID": "IC_ANOMALY_TEST_02",
        "Lot_ID": "LOT_001",
        "Value_0h_uA": 10.05,
        "Value_24h_uA": 16.50,  # +6.45 uA in 24h
        "PropagationDelay_0h_ns": 2.010,
        "PropagationDelay_24h_ns": 2.450,
        "Iddq_0h_mA": 1.20,
        "Iddq_24h_mA": 1.80,
        "Datasheet_Leakage_Limit_uA": 50.0,
    }

    res = pipeline.analyze_component(anomaly_data)

    assert res["risk_level"] == "HIGH_RISK"
    assert res["risk_score"] >= 75.0
    assert res["slope_exceeded"] == 1


# ==============================================================================
# Scenario 3: Latent Defect Below Datasheet Limit (Core SIH Objective)
# ==============================================================================
def test_scenario_3_latent_defect_below_datasheet_limit(pipeline):
    """
    Scenario 3: Latent defect that STAYS BELOW the 50 uA datasheet limit at 168h,
    escaping traditional screening, but is caught by our dynamic safety slope engine.
    """
    latent_defect_data = {
        "Component_ID": "IC_LATENT_ESCAPE_03",
        "Lot_ID": "LOT_001",
        "Value_0h_uA": 10.12,
        "Value_24h_uA": 11.56,  # 5x healthy drift rate
        "PropagationDelay_0h_ns": 2.015,
        "PropagationDelay_24h_ns": 2.218,
        "Iddq_0h_mA": 1.20,
        "Iddq_24h_mA": 1.35,
        "Datasheet_Leakage_Limit_uA": 50.0,
    }

    res = pipeline.analyze_component(latent_defect_data)

    # Must PASS static datasheet limit
    assert res["datasheet_pass"] is True
    assert res["predicted_168h"] < 50.0

    # Must BREACH safety slope and be classified as HIGH_RISK
    assert res["slope_exceeded"] == 1
    assert res["risk_level"] == "HIGH_RISK"
    assert res["risk_score"] >= 75.0
    assert "safety envelope" in str(res["explanation"]["human_readable_reasons"]).lower()


# ==============================================================================
# Scenario 4: Component Exceeding Datasheet Limit
# ==============================================================================
def test_scenario_4_component_exceeding_datasheet_limit(pipeline):
    """Scenario 4: Component with high baseline and massive drift crossing 50 uA."""
    failing_data = {
        "Component_ID": "IC_HARD_FAIL_04",
        "Lot_ID": "LOT_001",
        "Value_0h_uA": 38.00,
        "Value_24h_uA": 48.50,
        "PropagationDelay_0h_ns": 2.100,
        "PropagationDelay_24h_ns": 2.800,
        "Iddq_0h_mA": 2.50,
        "Iddq_24h_mA": 3.80,
        "Datasheet_Leakage_Limit_uA": 50.0,
    }

    res = pipeline.analyze_component(failing_data)

    assert res["predicted_168h"] > 50.0
    assert res["datasheet_pass"] is False
    assert res["risk_level"] == "HIGH_RISK"
    assert res["risk_score"] >= 80.0


# ==============================================================================
# Scenario 5: Missing Measurements
# ==============================================================================
def test_scenario_5_missing_measurements(pipeline):
    """Scenario 5: Telemetry dictionary missing optional channels; imputed without crash."""
    sparse_data = {
        "Component_ID": "IC_SPARSE_TEST_05",
        "Value_0h_uA": 10.15,
        "Value_24h_uA": 10.40,
        # Missing PropagationDelay, Iddq, Datasheet_Limit, Lot_ID
    }

    res = pipeline.analyze_component(sparse_data)

    assert res["component_id"] == "IC_SPARSE_TEST_05"
    assert res["risk_level"] == "NORMAL"
    assert res["predicted_168h"] > 0.0
    assert "explanation" in res


# ==============================================================================
# Scenario 6: Extreme / Corrupted Measurement
# ==============================================================================
def test_scenario_6_extreme_measurement(pipeline):
    """Scenario 6: Out-of-bounds sensor fault (negative value and gross surge > 500 uA)."""
    corrupted_data = {
        "Component_ID": "IC_FAULT_SENSOR_06",
        "Value_0h_uA": -15.00,  # Negative sensor error
        "Value_24h_uA": 1200.0,  # Massive 1200 uA surge
    }

    res = pipeline.analyze_component(corrupted_data)

    assert res["risk_level"] == "HIGH_RISK"
    assert res["risk_score"] == 100.0
    assert len(res["validation_warnings"]) >= 1
    assert any("surge" in w.lower() or "negative" in w.lower() for w in res["validation_warnings"])


# ==============================================================================
# High-Level Function API Test
# ==============================================================================
def test_analyze_component_high_level_api():
    """Verify top-level analyze_component function API."""
    data = {"Value_0h_uA": 10.0, "Value_24h_uA": 10.3}
    res = analyze_component(data)

    assert "component_id" in res
    assert "anomaly_score" in res
    assert "predicted_168h" in res
    assert "predicted_drift" in res
    assert "safety_slope" in res
    assert "risk_score" in res
    assert "risk_level" in res
    assert "explanation" in res
