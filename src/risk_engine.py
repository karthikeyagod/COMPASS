"""
Component Risk Engine Module for SIH Problem Statement 260170.

Integrates diagnostic and prognostic screening outputs:
- Module A: Dynamic Anomaly Score [0.0 - 1.0]
- Module B: 168-Hour Drift Projection (Value_168h_uA)
- Safety Slope: Trajectory Slope & Boundary Violation Index
- Batch / Peer Deviation: Manufacturing Lot Robust Outlier Metric

Generates:
- Composite Risk Score: [0.0 - 100.0]
- Risk Level: 'NORMAL', 'WATCH', or 'HIGH_RISK'
- Transparent sub-score decomposition and actionable engineering recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class RiskEngine:
    """
    Unified multi-factor decision engine for ESS component qualification.
    """

    DEFAULT_CONFIG_PATH = Path("models/risk_config.json")

    def __init__(self, config: Optional[Union[Dict[str, Any], Path, str]] = None) -> None:
        """
        Initialize RiskEngine with configuration dictionary or JSON filepath.

        Parameters
        ----------
        config : Optional[Union[Dict[str, Any], Path, str]]
            Configuration dictionary or path to configuration file.
        """
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        elif isinstance(config, dict):
            self.config = config
        elif self.DEFAULT_CONFIG_PATH.exists():
            with open(self.DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = {
                "weights": {
                    "anomaly_weight": 0.35,
                    "drift_projection_weight": 0.30,
                    "safety_slope_weight": 0.25,
                    "batch_deviation_weight": 0.10,
                },
                "risk_thresholds": {
                    "normal_max": 35.0,
                    "watch_max": 70.0,
                },
                "escalation_rules": {
                    "enable_critical_escalation": True,
                    "slope_breach_anomaly_threshold": 0.50,
                    "min_high_risk_score": 75.0,
                },
                "sub_score_normalization": {
                    "batch_z_score_saturation_limit": 4.0,
                    "slope_ratio_saturation_limit": 2.0,
                },
            }

        self.weights = self.config.get("weights", {})
        self.w_anomaly = self.weights.get("anomaly_weight", 0.35)
        self.w_drift = self.weights.get("drift_projection_weight", 0.30)
        self.w_slope = self.weights.get("safety_slope_weight", 0.25)
        self.w_batch = self.weights.get("batch_deviation_weight", 0.10)

        # Normalize weights to sum to 1.0
        total_w = self.w_anomaly + self.w_drift + self.w_slope + self.w_batch
        if total_w > 0:
            self.w_anomaly /= total_w
            self.w_drift /= total_w
            self.w_slope /= total_w
            self.w_batch /= total_w

        self.thresholds = self.config.get("risk_thresholds", {"normal_max": 35.0, "watch_max": 70.0})
        self.escalation = self.config.get("escalation_rules", {"enable_critical_escalation": True})

    def compute_sub_scores(
        self,
        anomaly_score: float,
        predicted_168h: float,
        predicted_drift_rate: float,
        safety_slope: float,
        datasheet_limit: float = 50.0,
        value_0h: float = 10.0,
        batch_deviation: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate individual transparent sub-scores normalized to a [0.0 - 100.0] scale.

        Parameters
        ----------
        anomaly_score : float
            Module A normalized anomaly score [0.0 - 1.0].
        predicted_168h : float
            Module B predicted 168h leakage current in uA.
        predicted_drift_rate : float
            Projected degradation slope (uA/hour).
        safety_slope : float
            Safety slope threshold boundary (uA/hour).
        datasheet_limit : float
            Static upper-specification limit (uA). Default 50.0.
        value_0h : float
            Baseline 0h leakage measurement (uA).
        batch_deviation : float
            Intra-lot robust Z-score magnitude.

        Returns
        -------
        Dict[str, float]
            Dictionary of sub-scores on [0 - 100] scale.
        """
        eps = 1e-7

        # 1. Anomaly Sub-Score: Direct scaling from Module A [0 - 100]
        s_anomaly = float(np.clip(anomaly_score * 100.0, 0.0, 100.0))

        # 2. Drift Projection Sub-Score: Relative consumption of allowable datasheet headroom
        headroom = max(eps, datasheet_limit - value_0h)
        projected_rise = max(0.0, predicted_168h - value_0h)
        drift_fraction = projected_rise / headroom
        s_drift = float(np.clip(drift_fraction * 100.0, 0.0, 100.0))

        # 3. Safety Slope Sub-Score: Ratio of predicted slope to safety boundary
        # Slope ratio = 1.0 -> 50 points; Slope ratio >= 2.0 -> 100 points
        slope_ratio = max(0.0, predicted_drift_rate) / max(eps, safety_slope)
        s_slope = float(np.clip(slope_ratio * 50.0, 0.0, 100.0))

        # 4. Batch Deviation Sub-Score: Standardized distance from manufacturing lot median
        z_saturation = self.config.get("sub_score_normalization", {}).get("batch_z_score_saturation_limit", 4.0)
        s_batch = float(np.clip((abs(batch_deviation) / max(eps, z_saturation)) * 100.0, 0.0, 100.0))

        return {
            "anomaly_sub_score": round(s_anomaly, 2),
            "drift_sub_score": round(s_drift, 2),
            "slope_sub_score": round(s_slope, 2),
            "batch_sub_score": round(s_batch, 2),
            "slope_ratio": round(slope_ratio, 4),
        }

    def evaluate_component(
        self,
        anomaly_score: float,
        predicted_168h: float,
        predicted_drift_rate: float,
        safety_slope: float,
        datasheet_limit: float = 50.0,
        value_0h: float = 10.0,
        batch_deviation: float = 0.0,
        component_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single component, compute composite Risk Score, and categorize Risk Level.

        Returns
        -------
        Dict[str, Any]
            Complete assessment report with sub-scores, verdicts, and recommendation.
        """
        sub = self.compute_sub_scores(
            anomaly_score=anomaly_score,
            predicted_168h=predicted_168h,
            predicted_drift_rate=predicted_drift_rate,
            safety_slope=safety_slope,
            datasheet_limit=datasheet_limit,
            value_0h=value_0h,
            batch_deviation=batch_deviation,
        )

        # Base Weighted Risk Score
        base_score = (
            (self.w_anomaly * sub["anomaly_sub_score"]) +
            (self.w_drift * sub["drift_sub_score"]) +
            (self.w_slope * sub["slope_sub_score"]) +
            (self.w_batch * sub["batch_sub_score"])
        )

        # Domain Boolean Verdicts
        datasheet_pass = bool(predicted_168h <= datasheet_limit)
        safety_slope_breached = bool(predicted_drift_rate > safety_slope)
        dynamic_anomaly_flag = bool(anomaly_score >= 0.334)  # Module A calibrated boundary
        batch_outlier_flag = bool(abs(batch_deviation) >= 2.5)

        # Safety-Critical Escalation Logic (Core SIH Requirement)
        # If Safety Slope is breached AND Dynamic Anomaly is elevated, enforce HIGH_RISK even if below datasheet limit
        final_score = base_score
        escalated = False
        if self.escalation.get("enable_critical_escalation", True):
            anomaly_gate = self.escalation.get("slope_breach_anomaly_threshold", 0.50)
            min_high_risk = self.escalation.get("min_high_risk_score", 75.0)

            if safety_slope_breached and (anomaly_score >= anomaly_gate or sub["slope_ratio"] >= 1.5):
                final_score = max(final_score, min_high_risk)
                escalated = True

        final_score = float(np.clip(final_score, 0.0, 100.0))

        # Categorize Risk Level
        normal_max = self.thresholds.get("normal_max", 35.0)
        watch_max = self.thresholds.get("watch_max", 70.0)

        if final_score < normal_max:
            risk_level = "NORMAL"
            recommendation = "PASS: Nominal degradation within healthy statistical envelope. Approved for flight/deployment."
        elif final_score < watch_max:
            risk_level = "WATCH"
            recommendation = "MONITOR: Moderate parametric drift or peer outlier tendency. Recommended for extended 48h screening."
        else:
            risk_level = "HIGH_RISK"
            if datasheet_pass and safety_slope_breached:
                recommendation = (
                    "REJECT / QUARANTINE: Component passes static datasheet limit but exhibits hazardous dynamic anomaly "
                    "and non-linear slope breach. High risk of in-field latent defect escape."
                )
            else:
                recommendation = "REJECT / QUARANTINE: Severe degradation trajectory or datasheet limit violation."

        result = {
            "component_id": component_id,
            "risk_score": round(final_score, 2),
            "risk_level": risk_level,
            "base_score_unscaled": round(base_score, 2),
            "escalated_for_safety": escalated,
            "sub_scores": {
                "anomaly_sub_score": sub["anomaly_sub_score"],
                "drift_sub_score": sub["drift_sub_score"],
                "slope_sub_score": sub["slope_sub_score"],
                "batch_sub_score": sub["batch_sub_score"],
            },
            "verdicts": {
                "datasheet_pass": datasheet_pass,
                "dynamic_anomaly_flag": dynamic_anomaly_flag,
                "safety_slope_breached": safety_slope_breached,
                "batch_outlier_flag": batch_outlier_flag,
            },
            "inputs_summary": {
                "anomaly_score": round(float(anomaly_score), 4),
                "predicted_168h_uA": round(float(predicted_168h), 4),
                "predicted_drift_rate_uA_per_h": round(float(predicted_drift_rate), 6),
                "safety_slope_uA_per_h": round(float(safety_slope), 6),
                "datasheet_limit_uA": round(float(datasheet_limit), 2),
                "value_0h_uA": round(float(value_0h), 4),
                "batch_deviation_z": round(float(batch_deviation), 4),
            },
            "recommendation": recommendation,
        }

        return result

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        anomaly_scores: Union[pd.Series, np.ndarray],
        predicted_168h: Union[pd.Series, np.ndarray],
        safety_slope: float,
        datasheet_limit_col: str = "Datasheet_Leakage_Limit_uA",
        value_0h_col: str = "Value_0h_uA",
        batch_z_col: Optional[str] = "Value_24h_Robust_Z_Score",
    ) -> pd.DataFrame:
        """
        Evaluate full dataset in vectorized fashion.

        Parameters
        ----------
        df : pd.DataFrame
            Component DataFrame.
        anomaly_scores : Union[pd.Series, np.ndarray]
            Module A anomaly scores.
        predicted_168h : Union[pd.Series, np.ndarray]
            Module B 168h predictions.
        safety_slope : float
            Calibrated safety slope boundary.
        datasheet_limit_col : str
            Datasheet limit column.
        value_0h_col : str
            0h baseline column.
        batch_z_col : Optional[str]
            Lot robust z-score column.

        Returns
        -------
        pd.DataFrame
            DataFrame with appended Risk_Score, Risk_Level, and sub-score metrics.
        """
        v0 = df[value_0h_col].values if value_0h_col in df.columns else np.full(len(df), 10.0)
        limits = df[datasheet_limit_col].values if datasheet_limit_col in df.columns else np.full(len(df), 50.0)
        batch_z = df[batch_z_col].values if (batch_z_col and batch_z_col in df.columns) else np.zeros(len(df))
        
        ano_arr = np.asarray(anomaly_scores, dtype=float)
        p168_arr = np.asarray(predicted_168h, dtype=float)
        drift_rates = (p168_arr - v0) / 168.0

        records = []
        for i in range(len(df)):
            cid = df["Component_ID"].iloc[i] if "Component_ID" in df.columns else f"IC_{i:06d}"
            res = self.evaluate_component(
                anomaly_score=ano_arr[i],
                predicted_168h=p168_arr[i],
                predicted_drift_rate=drift_rates[i],
                safety_slope=safety_slope,
                datasheet_limit=limits[i],
                value_0h=v0[i],
                batch_deviation=batch_z[i],
                component_id=cid,
            )
            records.append({
                "Risk_Score": res["risk_score"],
                "Risk_Level": res["risk_level"],
                "Anomaly_Sub_Score": res["sub_scores"]["anomaly_sub_score"],
                "Drift_Sub_Score": res["sub_scores"]["drift_sub_score"],
                "Slope_Sub_Score": res["sub_scores"]["slope_sub_score"],
                "Batch_Sub_Score": res["sub_scores"]["batch_sub_score"],
                "Datasheet_Pass": res["verdicts"]["datasheet_pass"],
                "Safety_Slope_Breached": res["verdicts"]["safety_slope_breached"],
                "Escalated_For_Safety": res["escalated_for_safety"],
            })

        df_res = pd.DataFrame(records, index=df.index)
        return pd.concat([df.copy(), df_res], axis=1)
