"""
Tests for the Secure & Trustworthy ESS Analytics Trust Layer.
Covers Cryptographic SHA-256 Integrity, Schema Fingerprinting,
Tamper Detection, Statistical Poisoning Detection, Trust Gating,
Audit Trail Logging, and Zero Data Leakage.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.trust_layer import (
    AuditTrailLogger,
    DatasetIntegrityChecker,
    PoisoningDetector,
    SchemaFingerprinter,
    TamperDetector,
    TrustGate,
    TrustLayer,
)
from src.drift_prediction import DriftPredictor


@pytest.fixture
def sample_clean_df():
    """Fixture providing a clean, compliant ESS dataset slice."""
    return pd.DataFrame({
        "Component_ID": [f"IC_{i:04d}" for i in range(1, 51)],
        "Lot_ID": ["LOT_001"] * 25 + ["LOT_002"] * 25,
        "Value_0h_uA": np.random.normal(10.0, 0.4, 50),
        "Value_24h_uA": np.random.normal(10.5, 0.5, 50),
        "Value_96h_uA": np.random.normal(11.2, 0.6, 50),
        "Value_168h_uA": np.random.normal(11.8, 0.7, 50),
    })


# ==============================================================================
# Test 1: Deterministic SHA-256 Hash Generation
# ==============================================================================
def test_sha256_deterministic_generation(sample_clean_df):
    hash1 = DatasetIntegrityChecker.compute_sha256(sample_clean_df)
    hash2 = DatasetIntegrityChecker.compute_sha256(sample_clean_df)
    assert hash1 == hash2
    assert len(hash1) == 64
    assert isinstance(hash1, str)


# ==============================================================================
# Test 2: Tampering Detection via SHA-256 Mismatch
# ==============================================================================
def test_tampering_detection_sha256_mismatch(sample_clean_df):
    clean_hash = DatasetIntegrityChecker.compute_sha256(sample_clean_df)
    
    # Modify a single cell
    tampered_df = sample_clean_df.copy()
    tampered_df.loc[0, "Value_24h_uA"] = 99.99
    
    res = DatasetIntegrityChecker.verify_integrity(
        content=tampered_df,
        expected_hash=clean_hash,
    )
    assert res["status"] == "INTEGRITY_MISMATCH"
    assert res["is_valid"] is False
    assert res["current_hash"] != clean_hash


# ==============================================================================
# Test 3: Schema Fingerprinting & Column Alteration Detection
# ==============================================================================
def test_schema_fingerprinting_and_change_detection(sample_clean_df):
    schema_res_clean = SchemaFingerprinter.analyze_schema(sample_clean_df)
    assert schema_res_clean["status"] == "VALID"
    assert schema_res_clean["is_valid"] is True

    # Drop required ID column
    altered_df = sample_clean_df.drop(columns=["Component_ID"])
    schema_res_altered = SchemaFingerprinter.analyze_schema(altered_df)
    assert schema_res_altered["status"] == "SCHEMA_CHANGED"
    assert schema_res_altered["is_valid"] is False
    assert any("component" in iss.lower() for iss in schema_res_altered["issues"])


# ==============================================================================
# Test 4: Duplicate Component IDs Detection
# ==============================================================================
def test_duplicate_component_ids_detection(sample_clean_df):
    tampered_df = sample_clean_df.copy()
    tampered_df.loc[1, "Component_ID"] = tampered_df.loc[0, "Component_ID"]
    
    tamper_res = TamperDetector.check_tampering(tampered_df)
    assert tamper_res["status"] == "WARNING"
    assert tamper_res["is_passed"] is False
    assert tamper_res["duplicate_ids_count"] >= 1
    assert any("Duplicate" in iss for iss in tamper_res["issues"])


# ==============================================================================
# Test 5: Physically Impossible Values Detection
# ==============================================================================
def test_impossible_physical_values_detection(sample_clean_df):
    tampered_df = sample_clean_df.copy()
    tampered_df.loc[3, "Value_24h_uA"] = -15.50  # Impossible negative leakage current
    
    tamper_res = TamperDetector.check_tampering(tampered_df)
    assert tamper_res["status"] == "WARNING"
    assert tamper_res["is_passed"] is False
    assert any("negative values" in iss for iss in tamper_res["issues"])


# ==============================================================================
# Test 6: Statistical Poisoning Detection (Injected MAD Outliers)
# ==============================================================================
def test_statistical_poisoning_mad_outliers(sample_clean_df):
    poisoned_df = sample_clean_df.copy()
    # Inject 5 extreme outliers
    for idx in range(5):
        poisoned_df.loc[idx, "Value_24h_uA"] = 85.0 + idx * 5.0
    
    poison_res = PoisoningDetector.evaluate_poisoning_risk(poisoned_df)
    assert poison_res["extreme_outliers_count"] >= 3
    assert poison_res["risk_score"] > 0.30
    assert any("extreme statistical outliers" in ind for ind in poison_res["indicators"])


# ==============================================================================
# Test 7: Population Distribution Shift Detection (KS-Test)
# ==============================================================================
def test_distribution_shift_detection(sample_clean_df):
    shifted_df = sample_clean_df.copy()
    # Severely shift mean & variance
    shifted_df["Value_24h_uA"] = np.random.normal(35.0, 8.0, len(sample_clean_df))
    
    poison_res = PoisoningDetector.evaluate_poisoning_risk(
        df=shifted_df,
        baseline_df=sample_clean_df,
    )
    assert poison_res["distribution_shift"] in ["MODERATE", "HIGH"]
    assert any("distribution shift" in ind for ind in poison_res["indicators"])


# ==============================================================================
# Test 8: Clean Dataset Authorized by Trust Gate
# ==============================================================================
def test_clean_dataset_trust_gate_authorization(sample_clean_df):
    eval_res = TrustLayer.evaluate_dataset(
        df=sample_clean_df,
        dataset_id="TEST_CLEAN_DATASET",
        baseline_df=sample_clean_df,
    )
    assert eval_res["overall_trust"] == "TRUSTED"
    assert eval_res["ml_pipeline_allowed"] is True
    assert eval_res["tamper_detection"]["status"] == "PASSED"
    assert eval_res["poison_detection"]["risk_level"] == "LOW"
    assert "audit_record" in eval_res
    assert eval_res["audit_record"]["ml_pipeline_allowed"] is True


# ==============================================================================
# Test 9: Critical Trust Failure Withholds Automatic Safety Decision
# ==============================================================================
def test_critical_trust_failure_withholds_decision(sample_clean_df):
    clean_hash = DatasetIntegrityChecker.compute_sha256(sample_clean_df)
    
    corrupted_df = sample_clean_df.copy()
    corrupted_df["Value_24h_uA"] = np.random.normal(120.0, 20.0, len(sample_clean_df))
    
    eval_res = TrustLayer.evaluate_dataset(
        df=corrupted_df,
        expected_hash=clean_hash,  # Mismatch guaranteed
        baseline_df=sample_clean_df,
    )
    assert eval_res["overall_trust"] == "NOT_TRUSTED"
    assert eval_res["ml_pipeline_allowed"] is False
    assert "withheld" in eval_res["recommendation"].lower()


# ==============================================================================
# Test 10: Strict Module B Inference Leakage Audit
# ==============================================================================
def test_module_b_zero_leakage_preservation():
    """Verify Module B strictly forbids Value_168h in inference feature matrix."""
    from src.feature_engineering import ESSStage, STAGE_FEATURE_REGISTRY
    early_features = STAGE_FEATURE_REGISTRY[ESSStage.STAGE_24H]
    assert "Value_168h" not in early_features
    assert "Value_168h_uA" not in early_features
    assert "Value_96h" not in early_features
    assert "Value_96h_uA" not in early_features
    assert all("168" not in col for col in early_features)
    assert all("96" not in col for col in early_features)
