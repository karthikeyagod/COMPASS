"""
Unit tests for Explainability Module (Phase 10).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.explainability import ExplainabilityEngine, QA_FEATURE_DICTIONARY


@pytest.fixture
def explainability_engine() -> ExplainabilityEngine:
    """Load default ExplainabilityEngine with workspace models."""
    return ExplainabilityEngine.load_default()


@pytest.fixture
def mock_sample_component() -> pd.DataFrame:
    """Generate sample component data mimicking an escaping latent defect."""
    return pd.DataFrame([{
        "Component_ID": "IC_TEST_ESCAPE_001",
        "Lot_ID": "Lot_042",
        "Value_0h_uA": 10.12,
        "Value_24h_uA": 11.56,
        "Early_Drift_uA": 1.44,
        "Early_Drift_Rate_uA_per_h": 0.0600,
        "Early_Percentage_Change": 14.23,
        "PropagationDelay_0h_ns": 2.015,
        "PropagationDelay_24h_ns": 2.218,
        "Early_Delay_Drift_ns": 0.203,
        "Early_Delay_Drift_Rate_ns_per_h": 0.00846,
        "Early_Delay_Percentage_Change": 10.07,
        "Iddq_0h_mA": 1.20,
        "Iddq_24h_mA": 1.35,
        "Early_Iddq_Drift_mA": 0.15,
        "Early_Iddq_Percentage_Change": 12.5,
        "Value_24h_Robust_Z_Score": 3.48,
        "Early_Drift_Robust_Z_Score": 3.65,
        "Early_Drift_Rate_per_h_Robust_Z_Score": 3.65,
        "Early_Delay_Drift_Robust_Z_Score": 2.89,
        "Datasheet_Leakage_Limit_uA": 50.0,
    }])


def test_explainability_structure(explainability_engine, mock_sample_component):
    """Verify that explain_component returns all required structured fields."""
    exp = explainability_engine.explain_component(mock_sample_component)

    assert "component_id" in exp
    assert "risk_level" in exp
    assert "risk_score" in exp
    assert "top_contributing_features" in exp
    assert "human_readable_reasons" in exp
    assert "supporting_measurements" in exp
    assert "verdict_summary" in exp

    assert exp["component_id"] == "IC_TEST_ESCAPE_001"
    assert exp["risk_level"] == "HIGH_RISK"
    assert exp["risk_score"] >= 75.0
    assert len(exp["top_contributing_features"]) > 0
    assert len(exp["human_readable_reasons"]) == 4


def test_qa_feature_translations():
    """Verify that technical column names map to human-readable names."""
    assert "Value_24h_Robust_Z_Score" in QA_FEATURE_DICTIONARY
    assert "Early_Drift_Rate_uA_per_h" in QA_FEATURE_DICTIONARY
    assert "Early_Delay_Drift_ns" in QA_FEATURE_DICTIONARY

    disp = QA_FEATURE_DICTIONARY["Early_Drift_Rate_uA_per_h"]
    assert "Velocity" in disp or "Drift" in disp


def test_reasons_content(explainability_engine, mock_sample_component):
    """Verify that human-readable reasons cover the key SIH diagnostic criteria."""
    exp = explainability_engine.explain_component(mock_sample_component)
    reasons = exp["human_readable_reasons"]

    # 1. Early leakage increase
    assert any("leakage" in r.lower() for r in reasons)
    # 2. Lot peer deviation
    assert any("lot" in r.lower() or "z-score" in r.lower() for r in reasons)
    # 3. 168h forecast
    assert any("168h" in r.lower() or "predicted" in r.lower() for r in reasons)
    # 4. Safety slope breach
    assert any("safety" in r.lower() or "envelope" in r.lower() or "breach" in r.lower() for r in reasons)


def test_generate_text_report(explainability_engine, mock_sample_component):
    """Verify text report rendering."""
    exp = explainability_engine.explain_component(mock_sample_component)
    text_report = explainability_engine.generate_text_report(exp)

    assert "ESS DIAGNOSTIC QUALITY REPORT" in text_report
    assert "OVERALL RISK VERDICT : HIGH_RISK" in text_report
    assert "ROOT-CAUSE DIAGNOSTIC REASONS:" in text_report
    assert "TOP CONTRIBUTING RISK FEATURES" in text_report
    assert "SUPPORTING PHYSICAL TELEMETRY:" in text_report
    assert "QA RECOMMENDATION:" in text_report
