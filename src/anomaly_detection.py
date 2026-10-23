"""
Anomaly Detection Module (Module A) for SIH Problem Statement 260170.

Implements statistical and machine learning anomaly detection for early Environmental
Stress Screening (ESS) at t <= 24h.

Classes & Detectors:
1. ZScoreDetector: Classical Gaussian standard deviation filter (|z| > threshold).
2. RobustZScoreDetector: Median & normalized Interquartile Range (IQR) filter.
3. IQRDetector: Non-parametric Tukey's boxplot fence filter (Q3 + k * IQR).
4. DynamicAnomalyDetector: Unsupervised Isolation Forest model equipped with continuous
   anomaly scoring, validation threshold tuning, and model serialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.feature_engineering import (
    DataLeakageError,
    ESSStage,
    validate_feature_availability,
)


class ZScoreDetector:
    """
    Classical parametric Z-score anomaly detector.
    Identifies observations where |x - mean| / std > threshold.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        """
        Parameters
        ----------
        threshold : float
            Standard deviation multiplier cutoff (default 3.0).
        """
        self.threshold = threshold
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, X: Union[pd.Series, np.ndarray]) -> "ZScoreDetector":
        """Fit parametric mean and standard deviation from nominal training series."""
        values = np.asarray(X, dtype=float)
        self.mean_ = float(np.nanmean(values))
        self.std_ = float(np.nanstd(values, ddof=1))
        if self.std_ == 0.0:
            self.std_ = 1e-7
        return self

    def score(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Calculate continuous absolute Z-scores."""
        values = np.asarray(X, dtype=float)
        return np.abs(values - self.mean_) / self.std_

    def predict(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Binary anomaly prediction (1 = Anomaly, 0 = Nominal)."""
        return (self.score(X) > self.threshold).astype(int)


class RobustZScoreDetector:
    """
    Robust non-parametric Z-score anomaly detector using Median and Normalized IQR.
    Resistant to outlier masking in contaminated training batches.
    Formula: |x - median| / (IQR / 1.34898)
    """

    def __init__(self, threshold: float = 3.0) -> None:
        """
        Parameters
        ----------
        threshold : float
            Robust standard deviation multiplier cutoff (default 3.0).
        """
        self.threshold = threshold
        self.median_: float = 0.0
        self.iqr_: float = 1.0

    def fit(self, X: Union[pd.Series, np.ndarray]) -> "RobustZScoreDetector":
        """Fit median and normalized IQR from training series."""
        values = np.asarray(X, dtype=float)
        self.median_ = float(np.nanmedian(values))
        q75, q25 = np.nanpercentile(values, [75, 25])
        iqr = q75 - q25
        self.iqr_ = float(iqr / 1.34898) if iqr > 0 else float(np.nanstd(values, ddof=1))
        if self.iqr_ == 0.0:
            self.iqr_ = 1e-7
        return self

    def score(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Calculate continuous absolute robust Z-scores."""
        values = np.asarray(X, dtype=float)
        return np.abs(values - self.median_) / self.iqr_

    def predict(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Binary anomaly prediction (1 = Anomaly, 0 = Nominal)."""
        return (self.score(X) > self.threshold).astype(int)


class IQRDetector:
    """
    Non-parametric Tukey's Fence anomaly detector.
    Flags points exceeding Q3 + k * IQR or falling below Q1 - k * IQR.
    """

    def __init__(self, k: float = 1.5) -> None:
        """
        Parameters
        ----------
        k : float
            Tukey fence multiplier (default 1.5 for outliers, 3.0 for extreme outliers).
        """
        self.k = k
        self.q25_: float = 0.0
        self.q75_: float = 0.0
        self.iqr_: float = 1.0
        self.lower_fence_: float = 0.0
        self.upper_fence_: float = 0.0

    def fit(self, X: Union[pd.Series, np.ndarray]) -> "IQRDetector":
        """Compute Tukey upper and lower fences from training series."""
        values = np.asarray(X, dtype=float)
        self.q75_, self.q25_ = np.nanpercentile(values, [75, 25])
        self.iqr_ = self.q75_ - self.q25_
        self.lower_fence_ = self.q25_ - (self.k * self.iqr_)
        self.upper_fence_ = self.q75_ + (self.k * self.iqr_)
        return self

    def predict(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Binary anomaly prediction (1 = Anomaly, 0 = Nominal)."""
        values = np.asarray(X, dtype=float)
        return ((values < self.lower_fence_) | (values > self.upper_fence_)).astype(int)


class DynamicAnomalyDetector:
    """
    Module A: Dynamic Anomaly Detection engine using Isolation Forest.
    
    Trained on early screening features (t <= 24h) to isolate anomalous drift kinetics
    and peer-relative outliers without requiring ground-truth labels during training.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Hyperparameters for Isolation Forest (n_estimators, contamination, random_state).
        feature_names : Optional[List[str]]
            Curated list of early feature names to use for training and inference.
        """
        self.config = config or {
            "n_estimators": 150,
            "contamination": 0.085,  # Matches approximate prior defect rate
            "max_samples": "auto",
            "random_state": 42,
            "n_jobs": -1,
        }
        self.feature_names: List[str] = feature_names or []
        if self.feature_names:
            if "Latent_Defect" in self.feature_names:
                raise DataLeakageError("Target 'Latent_Defect' must not be included in anomaly feature set.")
            validate_feature_availability(self.feature_names, current_stage=ESSStage.STAGE_24H)

        self.model: Optional[IsolationForest] = None
        self.threshold: float = 0.50  # Default calibrated decision threshold
        self.score_min_: float = 0.0
        self.score_max_: float = 1.0

    def _validate_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate feature set against ground truth leakage and time-series rules."""
        if "Latent_Defect" in self.feature_names:
            raise DataLeakageError("Target 'Latent_Defect' must not be included in anomaly feature set.")
        
        validate_feature_availability(self.feature_names, current_stage=ESSStage.STAGE_24H)
        
        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

        return df[self.feature_names]

    def fit(self, df: pd.DataFrame) -> "DynamicAnomalyDetector":
        """
        Fit Isolation Forest model on early-stage feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered DataFrame containing early features.

        Returns
        -------
        DynamicAnomalyDetector
            Fitted detector instance.
        """
        if not self.feature_names:
            raise ValueError("feature_names list must be provided before fitting.")

        X = self._validate_inputs(df)

        self.model = IsolationForest(
            n_estimators=self.config.get("n_estimators", 150),
            contamination=self.config.get("contamination", 0.085),
            max_samples=self.config.get("max_samples", "auto"),
            random_state=self.config.get("random_state", 42),
            n_jobs=self.config.get("n_jobs", -1),
        )
        self.model.fit(X)

        # Fit min/max calibration for score normalization
        raw_scores = -self.model.score_samples(X)
        self.score_min_ = float(np.min(raw_scores))
        self.score_max_ = float(np.max(raw_scores))
        if self.score_max_ == self.score_min_:
            self.score_max_ += 1e-7

        return self

    def predict_anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute continuous normalized anomaly scores in [0, 1].
        Higher score indicates stronger anomaly likelihood.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered DataFrame.

        Returns
        -------
        np.ndarray
            1D array of normalized anomaly scores in range [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X = self._validate_inputs(df)
        raw_scores = -self.model.score_samples(X)
        
        # Min-max normalization mapped to [0, 1]
        norm_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_)
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary anomaly flags using decision threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered DataFrame.
        threshold : Optional[float]
            Custom threshold. Defaults to self.threshold.

        Returns
        -------
        np.ndarray
            Binary array (1 = Anomaly/Defect Flag, 0 = Nominal/Pass).
        """
        thresh = threshold if threshold is not None else self.threshold
        scores = self.predict_anomaly_scores(df)
        return (scores >= thresh).astype(int)

    def tune_threshold(
        self,
        df_val: pd.DataFrame,
        y_val: Union[pd.Series, np.ndarray],
        min_recall: float = 0.98,
        preference: str = "f1",
    ) -> float:
        """
        Calibrate decision threshold on validation set to guarantee high recall
        and eliminate catastrophic false negative escapes.

        Parameters
        ----------
        df_val : pd.DataFrame
            Validation feature DataFrame.
        y_val : Union[pd.Series, np.ndarray]
            Validation ground-truth binary labels (1 = Latent Defect, 0 = Healthy).
        min_recall : float
            Target minimum recall constraint (default 0.98).
        preference : str
            Optimization objective ('f1' or 'precision').

        Returns
        -------
        float
            Optimal calibrated anomaly score threshold.
        """
        scores = self.predict_anomaly_scores(df_val)
        y_true = np.asarray(y_val, dtype=int)

        precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
        
        best_threshold = 0.50
        best_f1 = -1.0
        best_precision = -1.0

        # Scan thresholds meeting minimum recall
        valid_indices = np.where(recalls[:-1] >= min_recall)[0]

        if len(valid_indices) > 0:
            for idx in valid_indices:
                p = precisions[idx]
                r = recalls[idx]
                f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

                if preference == "f1" and f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(thresholds[idx])
                elif preference == "precision" and p > best_precision:
                    best_precision = p
                    best_threshold = float(thresholds[idx])
        else:
            # Fallback to threshold yielding highest recall
            max_r_idx = int(np.argmax(recalls[:-1]))
            best_threshold = float(thresholds[max_r_idx])

        self.threshold = best_threshold
        return self.threshold

    def evaluate(
        self,
        df: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute full classification and ranking metrics on a dataset.
        """
        y_arr = np.asarray(y_true, dtype=int)
        scores = self.predict_anomaly_scores(df)
        thresh = threshold if threshold is not None else self.threshold
        y_pred = (scores >= thresh).astype(int)

        tp = int(np.sum((y_arr == 1) & (y_pred == 1)))
        tn = int(np.sum((y_arr == 0) & (y_pred == 0)))
        fp = int(np.sum((y_arr == 0) & (y_pred == 1)))
        fn = int(np.sum((y_arr == 1) & (y_pred == 0)))

        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        pr_auc = float(average_precision_score(y_arr, scores))
        roc_auc = float(roc_auc_score(y_arr, scores))

        return {
            "threshold": thresh,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "recall": round(recall, 6),
            "precision": round(precision, 6),
            "f1_score": round(f1, 6),
            "false_negative_rate_fnr": round(fnr, 6),
            "pr_auc": round(pr_auc, 6),
            "roc_auc": round(roc_auc, 6),
            "silent_escapes": fn,
            "caught_defects": tp,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize trained model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "feature_names": self.feature_names,
            "model": self.model,
            "threshold": self.threshold,
            "score_min_": self.score_min_,
            "score_max_": self.score_max_,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DynamicAnomalyDetector":
        """Deserialize trained model from disk."""
        payload = joblib.load(filepath)
        instance = cls(config=payload["config"], feature_names=payload["feature_names"])
        instance.model = payload["model"]
        instance.threshold = payload["threshold"]
        instance.score_min_ = payload["score_min_"]
        instance.score_max_ = payload["score_max_"]
        return instance
