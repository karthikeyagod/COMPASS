"""
Unit tests for Module A Dynamic Anomaly Detection (Phase 5).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    DynamicAnomalyDetector,
    IQRDetector,
    RobustZScoreDetector,
    ZScoreDetector,
)
from src.feature_engineering import DataLeakageError, FeatureEngineer


def test_statistical_detectors():
    """Verify mathematical behavior of Z-score, Robust Z-score, and IQR detectors."""
    # Data with 20 nominal values around 10.0 and one extreme outlier at 100.0
    data = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.0,
                     10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.0, 100.0])

    z_det = ZScoreDetector(threshold=3.0).fit(data)
    z_preds = z_det.predict(data)
    assert z_preds[-1] == 1  # Outlier flagged

    rob_det = RobustZScoreDetector(threshold=3.0).fit(data)
    rob_preds = rob_det.predict(data)
    assert rob_preds[-1] == 1
    assert rob_preds[0] == 0

    iqr_det = IQRDetector(k=1.5).fit(data)
    iqr_preds = iqr_det.predict(data)
    assert iqr_preds[-1] == 1
    assert iqr_preds[0] == 0


def test_dynamic_anomaly_detector_data_leakage():
    """Verify that passing target or retrospective features raises DataLeakageError."""
    with pytest.raises(DataLeakageError):
        DynamicAnomalyDetector(feature_names=["Value_0h_uA", "Latent_Defect"])

    with pytest.raises(DataLeakageError):
        DynamicAnomalyDetector(feature_names=["Value_0h_uA", "Value_168h_uA"])


def test_dynamic_anomaly_detector_fit_predict_save(tmp_path):
    """Test full fit, scoring, threshold tuning, and serialization lifecycle."""
    np.random.seed(42)
    # Synthetic dataset of 100 nominal and 10 anomaly samples
    n_nom = 100
    n_ano = 10

    nom_data = {
        "Value_0h_uA": np.random.normal(10.0, 1.0, n_nom),
        "Value_24h_uA": np.random.normal(10.2, 1.0, n_nom),
        "Early_Drift_Rate_uA_per_h": np.random.normal(0.01, 0.005, n_nom),
    }
    ano_data = {
        "Value_0h_uA": np.random.normal(10.0, 1.0, n_ano),
        "Value_24h_uA": np.random.normal(13.0, 1.0, n_ano),
        "Early_Drift_Rate_uA_per_h": np.random.normal(0.12, 0.02, n_ano),
    }

    df = pd.concat([pd.DataFrame(nom_data), pd.DataFrame(ano_data)], ignore_index=True)
    y = np.array([0] * n_nom + [1] * n_ano)

    feat_cols = ["Value_0h_uA", "Value_24h_uA", "Early_Drift_Rate_uA_per_h"]
    detector = DynamicAnomalyDetector(
        config={"n_estimators": 50, "contamination": 0.1, "random_state": 42},
        feature_names=feat_cols,
    )
    detector.fit(df)

    scores = detector.predict_anomaly_scores(df)
    assert len(scores) == len(df)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    # Anomaly scores should be significantly higher for defect class
    assert np.mean(scores[y == 1]) > np.mean(scores[y == 0])

    # Threshold tuning
    thresh = detector.tune_threshold(df, y, min_recall=0.90)
    preds = detector.predict(df)
    assert np.sum(preds[y == 1]) >= 9  # At least 90% recall

    # Serialization roundtrip
    model_file = tmp_path / "test_anomaly.pkl"
    detector.save(model_file)
    loaded_detector = DynamicAnomalyDetector.load(model_file)

    loaded_scores = loaded_detector.predict_anomaly_scores(df)
    np.testing.assert_allclose(scores, loaded_scores)
    assert loaded_detector.threshold == thresh
