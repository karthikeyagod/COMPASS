"""
Unit tests for Traditional ESS Baseline Module (Phase 4).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.baseline import evaluate_traditional_baseline, run_baseline_evaluation


@pytest.fixture
def mock_screening_data() -> pd.DataFrame:
    """Mock screening data with known defect and pass/fail states."""
    return pd.DataFrame({
        "Component_ID": ["IC_1", "IC_2", "IC_3", "IC_4", "IC_5"],
        "Value_168h_uA": [12.0, 48.0, 52.0, 15.0, 55.0],
        "Datasheet_Leakage_Limit_uA": [50.0, 50.0, 50.0, 50.0, 50.0],
        "Latent_Defect": [0, 1, 1, 0, 0],
    })


def test_baseline_metrics_computation(mock_screening_data):
    """
    Test evaluation logic:
    - IC_1: Val=12 <= 50 (Pred=0), True=0 -> TN
    - IC_2: Val=48 <= 50 (Pred=0), True=1 -> FN (Silent Escape)
    - IC_3: Val=52 > 50 (Pred=1), True=1 -> TP (Caught Defect)
    - IC_4: Val=15 <= 50 (Pred=0), True=0 -> TN
    - IC_5: Val=55 > 50 (Pred=1), True=0 -> FP (False Alarm)
    """
    metrics = evaluate_traditional_baseline(mock_screening_data)
    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] == 1
    assert cm["true_negatives"] == 2
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1

    cl = metrics["classification_metrics"]
    assert np.isclose(cl["accuracy"], 3.0 / 5.0)
    assert np.isclose(cl["precision"], 1.0 / 2.0)
    assert np.isclose(cl["recall"], 1.0 / 2.0)
    assert np.isclose(cl["false_negative_rate_fnr"], 1.0 / 2.0)


def test_baseline_on_actual_dataset():
    """Verify baseline evaluation on full synthetic dataset yields 100% silent escape."""
    dataset_path = Path("data/sih_260170_synthetic_ess_dataset.csv")
    if not dataset_path.exists():
        pytest.skip("Dataset file not present in workspace")

    df = pd.read_csv(dataset_path)
    metrics = evaluate_traditional_baseline(df)
    
    cm = metrics["confusion_matrix"]
    assert cm["true_positives"] == 0
    assert cm["false_negatives"] == 1024
    assert cm["true_negatives"] == 10976
    assert cm["false_positives"] == 0

    cl = metrics["classification_metrics"]
    assert cl["recall"] == 0.0
    assert cl["precision"] == 0.0
    assert cl["f1_score"] == 0.0
    assert cl["false_negative_rate_fnr"] == 1.0

    sih = metrics["critical_sih_analysis"]
    assert sih["latent_defects_missed_silent_escapes"] == 1024
    assert sih["silent_escape_rate_percent"] == 100.0
