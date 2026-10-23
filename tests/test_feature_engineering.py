"""
Unit tests for Feature Engineering Module (Phase 3).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    DataLeakageError,
    ESSStage,
    FeatureEngineer,
    compute_batch_relative_features,
    compute_early_drift_features,
    compute_retrospective_features,
    validate_feature_availability,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a deterministic synthetic sample DataFrame."""
    return pd.DataFrame({
        "Component_ID": ["IC_01", "IC_02", "IC_03", "IC_04"],
        "Lot_ID": ["LOT_A", "LOT_A", "LOT_B", "LOT_B"],
        "Temperature_C": [125, 125, 125, 125],
        "Value_0h_uA": [10.0, 10.0, 20.0, 20.0],
        "Value_24h_uA": [12.4, 10.0, 22.4, 24.8],
        "Value_96h_uA": [16.0, 11.0, 26.0, 30.0],
        "Value_168h_uA": [25.0, 12.0, 35.0, 45.0],
        "Iddq_0h_mA": [5.0, 5.0, 6.0, 6.0],
        "Iddq_24h_mA": [5.5, 5.0, 6.2, 6.6],
        "PropagationDelay_0h_ns": [2.0, 2.0, 2.5, 2.5],
        "PropagationDelay_24h_ns": [2.24, 2.0, 2.6, 2.8],
        "PropagationDelay_96h_ns": [2.6, 2.1, 2.9, 3.2],
        "PropagationDelay_168h_ns": [3.2, 2.2, 3.5, 4.0],
        "Latent_Defect": [1, 0, 0, 1],
        "Datasheet_Leakage_Limit_uA": [50.0, 50.0, 50.0, 50.0],
        "Absolute_Limit_Pass": [1, 1, 1, 1],
    })


def test_early_drift_computation(sample_df):
    """Test early drift formulas (Early_Drift, Early_Drift_Rate, Early_Percentage_Change)."""
    df_feat = compute_early_drift_features(sample_df)

    # IC_01: Value_0h = 10.0, Value_24h = 12.4
    # Early_Drift = 12.4 - 10.0 = 2.4 uA
    # Early_Drift_Rate = 2.4 / 24 = 0.1 uA/h
    # Early_Percentage_Change = (2.4 / 10.0) * 100 = 24.0%
    assert np.isclose(df_feat.loc[0, "Early_Drift_uA"], 2.4)
    assert np.isclose(df_feat.loc[0, "Early_Drift_Rate_uA_per_h"], 0.1)
    assert np.isclose(df_feat.loc[0, "Early_Percentage_Change"], 24.0)

    # IC_02: 0 drift
    assert np.isclose(df_feat.loc[1, "Early_Drift_uA"], 0.0)
    assert np.isclose(df_feat.loc[1, "Early_Drift_Rate_uA_per_h"], 0.0)
    assert np.isclose(df_feat.loc[1, "Early_Percentage_Change"], 0.0)

    # Delay drift for IC_01: 2.24 - 2.0 = 0.24 ns (0.01 ns/h, 12%)
    assert np.isclose(df_feat.loc[0, "Early_Delay_Drift_ns"], 0.24)
    assert np.isclose(df_feat.loc[0, "Early_Delay_Drift_Rate_ns_per_h"], 0.01)
    assert np.isclose(df_feat.loc[0, "Early_Delay_Percentage_Change"], 12.0)


def test_batch_relative_features(sample_df):
    """Test lot-level aggregation, robust z-score, and percentile ranks."""
    df_early = compute_early_drift_features(sample_df)
    df_lot, stats = compute_batch_relative_features(df_early, target_columns=["Value_24h_uA"])

    assert "Value_24h_Lot_Mean" in df_lot.columns
    assert "Value_24h_Lot_Median" in df_lot.columns
    assert "Value_24h_Lot_STD" in df_lot.columns
    assert "Value_24h_Lot_IQR" in df_lot.columns
    assert "Value_24h_Robust_Z_Score" in df_lot.columns
    assert "Value_24h_Lot_Percentile" in df_lot.columns

    # LOT_A has values 12.4 and 10.0 -> Mean = 11.2, Median = 11.2
    lot_a_rows = df_lot[df_lot["Lot_ID"] == "LOT_A"]
    assert np.isclose(lot_a_rows["Value_24h_Lot_Mean"].iloc[0], 11.2)
    assert np.isclose(lot_a_rows["Value_24h_Lot_Median"].iloc[0], 11.2)

    # Percentiles within LOT_A: 10.0 -> 50%, 12.4 -> 100%
    assert df_lot.loc[0, "Value_24h_Lot_Percentile"] == 100.0
    assert df_lot.loc[1, "Value_24h_Lot_Percentile"] == 50.0


def test_retrospective_features(sample_df):
    """Test retrospective formulas (96h, 168h total drift)."""
    df_ret = compute_retrospective_features(sample_df)

    # IC_01: 96h drift = 16.0 - 10.0 = 6.0 uA
    # 168h total drift = 25.0 - 10.0 = 15.0 uA (150% change)
    assert np.isclose(df_ret.loc[0, "Drift_96h_uA"], 6.0)
    assert np.isclose(df_ret.loc[0, "Total_Drift_uA"], 15.0)
    assert np.isclose(df_ret.loc[0, "Total_Percentage_Change"], 150.0)
    assert np.isclose(df_ret.loc[0, "Total_Drift_Rate_uA_per_h"], 15.0 / 168.0)


def test_data_leakage_enforcement():
    """Verify that using retrospective features at stage 24h raises DataLeakageError."""
    # Valid early features should pass without error
    valid_features = ["Value_0h_uA", "Value_24h_uA", "Early_Drift_uA", "Early_Drift_Rate_uA_per_h"]
    validate_feature_availability(valid_features, current_stage="24h")

    # Leaked retrospective raw features must raise DataLeakageError
    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        validate_feature_availability(["Value_0h_uA", "Value_96h_uA"], current_stage="24h")

    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        validate_feature_availability(["Value_0h_uA", "Value_168h_uA"], current_stage="24h")

    with pytest.raises(DataLeakageError, match="Data Leakage Violation"):
        validate_feature_availability(["Total_Drift_uA"], current_stage="24h")


def test_feature_engineer_pipeline(sample_df):
    """Verify stateful FeatureEngineer fit/transform and early feature extraction."""
    fe = FeatureEngineer(stage=ESSStage.STAGE_24H, include_batch_relative=True)
    df_transformed = fe.fit_transform(sample_df)

    early_feature_names = fe.get_early_feature_names(df_transformed)

    assert len(early_feature_names) > 0
    # Retrospective features must not be present in early_feature_names
    for retro in ["Value_96h_uA", "Value_168h_uA", "Total_Drift_uA", "Drift_96h_uA"]:
        assert retro not in early_feature_names


def test_full_dataset_feature_pipeline():
    """Run FeatureEngineer on actual SIH dataset to ensure robust scaling."""
    dataset_path = Path("data/sih_260170_synthetic_ess_dataset.csv")
    if not dataset_path.exists():
        pytest.skip("Dataset file not present in environment")

    df = pd.read_csv(dataset_path)
    fe = FeatureEngineer(stage=ESSStage.STAGE_24H, include_batch_relative=True)
    df_transformed = fe.fit_transform(df)

    assert len(df_transformed) == 12000
    early_features = fe.get_early_feature_names(df_transformed)
    assert "Early_Drift_Rate_uA_per_h" in early_features
    assert "Value_24h_Robust_Z_Score" in early_features
    assert "Value_168h_uA" not in early_features
