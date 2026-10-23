"""
Unit tests for Module A Threshold & False-Negative Optimization (Phase 6).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from training.optimize_threshold import (
    compute_threshold_sweep,
    run_threshold_optimization,
    select_optimal_threshold,
)


def test_compute_threshold_sweep():
    """Verify threshold sweep monotonicity and metric computation."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    thresholds = np.array([0.05, 0.35, 0.65, 0.95])

    sweep_df = compute_threshold_sweep(y_true, scores, thresholds)

    assert len(sweep_df) == 4
    # At t=0.05 (low threshold): Recall should be 1.0, FP should be 4
    assert sweep_df.loc[0, "Recall"] == 1.0
    assert sweep_df.loc[0, "False_Positives"] == 4

    # At t=0.95 (high threshold): Recall should be 0.0, FP should be 0
    assert sweep_df.loc[3, "Recall"] == 0.0
    assert sweep_df.loc[3, "False_Positives"] == 0

    # Monotonic decrease in False Positives as threshold increases
    assert np.all(np.diff(sweep_df["False_Positives"]) <= 0)


def test_select_optimal_threshold():
    """Verify safety-critical threshold selection rule."""
    sweep_data = pd.DataFrame({
        "Threshold": [0.2, 0.3, 0.4, 0.5, 0.6],
        "Recall": [1.0, 0.99, 0.95, 0.90, 0.70],
        "Precision": [0.60, 0.90, 0.94, 0.96, 0.98],
        "F1_Score": [0.75, 0.94, 0.945, 0.93, 0.82],
        "False_Positive_Rate": [0.05, 0.015, 0.008, 0.005, 0.001],
        "Asymmetric_Cost": [50.0, 15.0, 508.0, 1005.0, 3001.0],
    })

    thresh, row, rationale = select_optimal_threshold(
        sweep_data,
        min_recall_target=0.99,
        max_fpr_target=0.02,
    )

    # Should select t=0.3 which has Recall=0.99, FPR=0.015 <= 0.02, and F1=0.94
    assert thresh == 0.3
    assert row["Recall"] >= 0.99
    assert row["False_Positive_Rate"] <= 0.02
