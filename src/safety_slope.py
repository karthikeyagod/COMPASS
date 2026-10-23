"""
Safety Slope & Healthy Degradation Envelope Module for SIH Problem Statement 260170.

Implements data-driven safety slope boundary estimation from healthy training components,
evaluates whether predicted component degradation trajectories are physically permissible,
and calculates component-level safety margins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd


class SafetySlopeEstimator:
    """
    Estimates acceptable degradation slope boundaries from healthy population kinetics (t=168h)
    and evaluates incoming component degradation trajectories.
    """

    SUPPORTED_METHODS = [
        "robust_tukey",
        "percentile_99_7",
        "percentile_99",
        "gaussian_3sigma",
    ]

    def __init__(
        self,
        method: str = "robust_tukey",
        custom_slope: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        method : str
            Statistical boundary estimation method ('robust_tukey', 'percentile_99_7',
            'percentile_99', 'gaussian_3sigma'). Default: 'robust_tukey'.
        custom_slope : Optional[float]
            Manual override for safety slope boundary (uA/hour).
        """
        if method not in self.SUPPORTED_METHODS and custom_slope is None:
            raise ValueError(f"Unsupported method '{method}'. Choose from {self.SUPPORTED_METHODS}")

        self.method = method
        self.custom_slope = custom_slope
        self.safety_slope_: float = 0.0
        self.boundary_candidates_: Dict[str, float] = {}
        self.healthy_stats_: Dict[str, float] = {}
        self.healthy_trajectory_envelope_: Dict[int, Dict[str, float]] = {}

    def fit(
        self,
        df_healthy: pd.DataFrame,
        value_0h_col: str = "Value_0h_uA",
        value_168h_col: str = "Value_168h_uA",
        target_defect_col: Optional[str] = "Latent_Defect",
    ) -> "SafetySlopeEstimator":
        """
        Estimate empirical safety slope boundaries from healthy training observations.

        Parameters
        ----------
        df_healthy : pd.DataFrame
            DataFrame of training components (filtered to healthy ICs if Latent_Defect present).
        value_0h_col : str
            0h baseline measurement column name.
        value_168h_col : str
            168h final measurement column name.
        target_defect_col : Optional[str]
            Column indicating defect status for filtering.

        Returns
        -------
        SafetySlopeEstimator
            Fitted estimator instance.
        """
        df = df_healthy.copy()
        if target_defect_col and target_defect_col in df.columns:
            df = df[df[target_defect_col] == 0]

        if len(df) == 0:
            raise ValueError("No healthy components found to fit SafetySlopeEstimator.")

        # Compute empirical 168h degradation rate for healthy components
        drift = df[value_168h_col] - df[value_0h_col]
        drift_rates = (drift / 168.0).values  # uA / hour

        # Compute baseline statistical distribution
        mean_rate = float(np.mean(drift_rates))
        std_rate = float(np.std(drift_rates, ddof=1))
        median_rate = float(np.median(drift_rates))
        q25, q75 = np.percentile(drift_rates, [25, 75])
        iqr_rate = float(q75 - q25)
        p95 = float(np.percentile(drift_rates, 95))
        p99 = float(np.percentile(drift_rates, 99))
        p99_7 = float(np.percentile(drift_rates, 99.73))
        robust_tukey = float(q75 + (1.5 * iqr_rate))
        gaussian_3s = float(mean_rate + (3.0 * std_rate))

        self.healthy_stats_ = {
            "mean_drift_rate_uA_per_h": round(mean_rate, 6),
            "std_drift_rate_uA_per_h": round(std_rate, 6),
            "median_drift_rate_uA_per_h": round(median_rate, 6),
            "iqr_drift_rate_uA_per_h": round(iqr_rate, 6),
            "p95_drift_rate_uA_per_h": round(p95, 6),
            "p99_drift_rate_uA_per_h": round(p99, 6),
            "p99_7_drift_rate_uA_per_h": round(p99_7, 6),
            "robust_tukey_boundary_uA_per_h": round(robust_tukey, 6),
            "gaussian_3sigma_boundary_uA_per_h": round(gaussian_3s, 6),
        }

        self.boundary_candidates_ = {
            "robust_tukey": robust_tukey,
            "percentile_99_7": p99_7,
            "percentile_99": p99,
            "gaussian_3sigma": gaussian_3s,
        }

        if self.custom_slope is not None:
            self.safety_slope_ = float(self.custom_slope)
        else:
            self.safety_slope_ = float(self.boundary_candidates_[self.method])

        # Compute trajectory corridor across time points
        time_cols = [("Value_0h_uA", 0), ("Value_24h_uA", 24), ("Value_96h_uA", 96), ("Value_168h_uA", 168)]
        for col_name, t_hr in time_cols:
            if col_name in df.columns:
                series = df[col_name]
                t_mean = float(series.mean())
                t_std = float(series.std(ddof=1))
                t_q75 = float(series.quantile(0.75))
                t_iqr = float(series.quantile(0.75) - series.quantile(0.25))
                self.healthy_trajectory_envelope_[t_hr] = {
                    "mean": t_mean,
                    "median": float(series.median()),
                    "upper_3sigma": t_mean + (3.0 * t_std),
                    "upper_tukey": t_q75 + (1.5 * t_iqr),
                    "p99_7": float(np.percentile(series, 99.73)),
                }

        return self

    def evaluate_components(
        self,
        df: pd.DataFrame,
        predicted_168h: Union[pd.Series, np.ndarray],
        value_0h_col: str = "Value_0h_uA",
    ) -> pd.DataFrame:
        """
        Evaluate individual component trajectories against the safety slope boundary.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing baseline measurements.
        predicted_168h : Union[pd.Series, np.ndarray]
            Predicted 168h continuous leakage values (from Module B).
        value_0h_col : str
            Column name for 0h baseline measurement.

        Returns
        -------
        pd.DataFrame
            Evaluation DataFrame containing:
            - Predicted_168h
            - Predicted_Drift
            - Predicted_Drift_Rate
            - Safety_Slope
            - Slope_Exceeded
            - Safety_Margin
            - Slope_Violation_Ratio
        """
        if self.safety_slope_ == 0.0 and self.custom_slope is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        df_res = pd.DataFrame(index=df.index)
        
        # 1. Base identifiers if present
        if "Component_ID" in df.columns:
            df_res["Component_ID"] = df["Component_ID"]
        if "Lot_ID" in df.columns:
            df_res["Lot_ID"] = df["Lot_ID"]
        if "Latent_Defect" in df.columns:
            df_res["Latent_Defect"] = df["Latent_Defect"]
        if "Value_168h_uA" in df.columns:
            df_res["Actual_168h_uA"] = df["Value_168h_uA"]

        # 2. Safety Slope Calculations
        v0 = df[value_0h_col].values
        p168 = np.asarray(predicted_168h, dtype=float)

        pred_drift = p168 - v0
        pred_drift_rate = pred_drift / 168.0  # uA / hour
        slope_exceeded = (pred_drift_rate > self.safety_slope_).astype(int)
        safety_margin = self.safety_slope_ - pred_drift_rate
        violation_ratio = pred_drift_rate / max(1e-7, self.safety_slope_)

        df_res["Value_0h_uA"] = v0
        df_res["Predicted_168h"] = np.round(p168, 4)
        df_res["Predicted_Drift"] = np.round(pred_drift, 4)
        df_res["Predicted_Drift_Rate"] = np.round(pred_drift_rate, 6)
        df_res["Safety_Slope"] = np.round(self.safety_slope_, 6)
        df_res["Slope_Exceeded"] = slope_exceeded
        df_res["Safety_Margin"] = np.round(safety_margin, 6)
        df_res["Slope_Violation_Ratio"] = np.round(violation_ratio, 4)

        return df_res

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted safety slope estimator to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "custom_slope": self.custom_slope,
            "safety_slope_": self.safety_slope_,
            "boundary_candidates_": self.boundary_candidates_,
            "healthy_stats_": self.healthy_stats_,
            "healthy_trajectory_envelope_": self.healthy_trajectory_envelope_,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SafetySlopeEstimator":
        """Deserialize safety slope estimator from disk."""
        payload = joblib.load(filepath)
        instance = cls(method=payload["method"], custom_slope=payload["custom_slope"])
        instance.safety_slope_ = payload["safety_slope_"]
        instance.boundary_candidates_ = payload["boundary_candidates_"]
        instance.healthy_stats_ = payload["healthy_stats_"]
        instance.healthy_trajectory_envelope_ = payload["healthy_trajectory_envelope_"]
        return instance
