"""
Unit tests for Component Risk Engine Module (Phase 9).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.risk_engine import RiskEngine


@pytest.fixture
def risk_engine() -> RiskEngine:
    """Initialize RiskEngine with standard configuration."""
    return RiskEngine()


def test_nominal_component_risk(risk_engine):
    """Verify nominal healthy component receives low risk score and NORMAL level."""
    res = risk_engine.evaluate_component(
        anomaly_score=0.08,
        predicted_168h=12.2,
        predicted_drift_rate=0.012,
        safety_slope=0.037,
        datasheet_limit=50.0,
        value_0h=10.1,
        batch_deviation=0.2,
        component_id="IC_HEALTHY_01",
    )

    assert res["risk_score"] < 35.0
    assert res["risk_level"] == "NORMAL"
    assert res["verdicts"]["datasheet_pass"] is True
    assert res["verdicts"]["safety_slope_breached"] is False


def test_watch_component_risk(risk_engine):
    """Verify component with moderate drift receives intermediate WATCH level."""
    res = risk_engine.evaluate_component(
        anomaly_score=0.45,
        predicted_168h=22.0,
        predicted_drift_rate=0.035,
        safety_slope=0.037,
        datasheet_limit=50.0,
        value_0h=10.0,
        batch_deviation=1.8,
        component_id="IC_MODERATE_02",
    )

    assert 35.0 <= res["risk_score"] < 70.0
    assert res["risk_level"] == "WATCH"


def test_core_sih_latent_defect_scenario(risk_engine):
    """
    Verify the primary SIH problem statement scenario:
    - Anomaly Score: 0.91
    - Predicted 168h: 35.0 uA (Absolute limit 50.0 uA is NOT exceeded -> PASS)
    - Safety Slope: 0.03707 uA/h (Predicted drift rate = (35-10)/168 = 0.1488 uA/h -> EXCEEDED)
    - Batch Deviation: High (3.5)
    -> Result must be HIGH_RISK with Risk Score >= 80!
    """
    res = risk_engine.evaluate_component(
        anomaly_score=0.91,
        predicted_168h=35.0,
        predicted_drift_rate=(35.0 - 10.0) / 168.0,  # 0.1488 uA/h (4x safety slope)
        safety_slope=0.037074,
        datasheet_limit=50.0,
        value_0h=10.0,
        batch_deviation=3.5,
        component_id="IC_SIH_LATENT_ESCAPE",
    )

    assert res["verdicts"]["datasheet_pass"] is True, "Must pass static datasheet limit"
    assert res["verdicts"]["safety_slope_breached"] is True, "Must breach dynamic safety slope"
    assert res["verdicts"]["dynamic_anomaly_flag"] is True, "Must trigger dynamic anomaly"
    assert res["risk_score"] >= 80.0, f"Expected high risk score >= 80, got {res['risk_score']}"
    assert res["risk_level"] == "HIGH_RISK", f"Expected HIGH_RISK, got {res['risk_level']}"


def test_risk_config_file_loading():
    """Verify loading from models/risk_config.json."""
    config_path = Path("models/risk_config.json")
    assert config_path.exists(), "risk_config.json must exist"

    engine = RiskEngine(config=config_path)
    assert engine.w_anomaly > 0
    assert engine.thresholds["normal_max"] == 35.0
    assert engine.thresholds["watch_max"] == 70.0


def test_evaluate_dataframe(risk_engine):
    """Verify vectorized DataFrame batch risk evaluation."""
    df = pd.DataFrame({
        "Component_ID": ["IC_01", "IC_02", "IC_03"],
        "Value_0h_uA": [10.0, 10.0, 10.0],
        "Datasheet_Leakage_Limit_uA": [50.0, 50.0, 50.0],
        "Value_24h_Robust_Z_Score": [0.1, 1.5, 3.8],
    })
    ano_scores = np.array([0.05, 0.40, 0.95])
    p168_vals = np.array([12.0, 20.0, 38.0])

    res_df = risk_engine.evaluate_dataframe(
        df=df,
        anomaly_scores=ano_scores,
        predicted_168h=p168_vals,
        safety_slope=0.037,
    )

    assert "Risk_Score" in res_df.columns
    assert "Risk_Level" in res_df.columns
    assert res_df.loc[0, "Risk_Level"] == "NORMAL"
    assert res_df.loc[2, "Risk_Level"] == "HIGH_RISK"
