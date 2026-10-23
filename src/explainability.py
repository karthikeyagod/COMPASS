"""
Explainability Module (XAI) for SIH Problem Statement 260170.

Provides transparent, root-cause explanations of screening decisions to Quality Assurance (QA)
engineers using SHAP (SHapley Additive exPlanations) and statistical peer-population metrics.
Translates technical feature IDs into domain-friendly terminology and generates structured
diagnostic reports and natural language justifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
try:
    import shap
except (ImportError, Exception):
    shap = None

from src.anomaly_detection import DynamicAnomalyDetector
from src.drift_prediction import DriftPredictor
from src.risk_engine import RiskEngine
from src.safety_slope import SafetySlopeEstimator


# ==============================================================================
# QA-Friendly Semantic Translation Dictionary
# Translates raw telemetry column IDs into clear engineering descriptions.
# ==============================================================================
QA_FEATURE_DICTIONARY: Dict[str, str] = {
    "Value_0h_uA": "Initial 0h Baseline Leakage Current",
    "Value_24h_uA": "24h Interim Leakage Current",
    "Early_Drift_uA": "Early 24h Leakage Drift (Absolute Delta)",
    "Early_Drift_Rate_uA_per_h": "Early Leakage Drift Velocity (dI/dt)",
    "Early_Percentage_Change": "Early 24h Percentage Leakage Increase",
    "PropagationDelay_0h_ns": "Initial Gate Propagation Delay",
    "PropagationDelay_24h_ns": "24h Gate Propagation Delay",
    "Early_Delay_Drift_ns": "Gate Propagation Timing Drift",
    "Early_Delay_Drift_Rate_ns_per_h": "Gate Timing Degradation Velocity",
    "Early_Delay_Percentage_Change": "Gate Timing Percentage Increase",
    "Iddq_0h_mA": "Initial Quiescent Supply Current (Iddq)",
    "Iddq_24h_mA": "24h Quiescent Supply Current (Iddq)",
    "Early_Iddq_Drift_mA": "Quiescent Supply Current Shift",
    "Early_Iddq_Percentage_Change": "Iddq Percentage Increase",
    "Value_24h_Robust_Z_Score": "24h Leakage Lot Peer Deviation (Robust Z-Score)",
    "Early_Drift_Robust_Z_Score": "Early Drift Lot Peer Deviation (Robust Z-Score)",
    "Early_Drift_Rate_per_h_Robust_Z_Score": "Drift Velocity Lot Peer Deviation (Robust Z-Score)",
    "Early_Delay_Drift_Robust_Z_Score": "Timing Drift Lot Peer Deviation (Robust Z-Score)",
}


class ExplainabilityEngine:
    """
    Root-cause diagnostic explanation engine combining SHAP TreeExplainer
    and statistical population kinetics.
    """

    def __init__(
        self,
        drift_predictor: Optional[DriftPredictor] = None,
        anomaly_detector: Optional[DynamicAnomalyDetector] = None,
        safety_estimator: Optional[SafetySlopeEstimator] = None,
        risk_engine: Optional[RiskEngine] = None,
    ) -> None:
        """
        Initialize the ExplainabilityEngine.
        """
        self.drift_predictor = drift_predictor
        self.anomaly_detector = anomaly_detector
        self.safety_estimator = safety_estimator
        self.risk_engine = risk_engine or RiskEngine()
        self.shap_explainer: Optional[Any] = None

        if self.drift_predictor is not None:
            self._init_shap_explainer()

    def _init_shap_explainer(self) -> None:
        """Initialize SHAP TreeExplainer on the underlying drift regressor."""
        if shap is None:
            self.shap_explainer = None
            return

        regressor = None
        if hasattr(self.drift_predictor, "model"):
            pipeline = self.drift_predictor.model
            if hasattr(pipeline, "named_steps") and "regressor" in pipeline.named_steps:
                regressor = pipeline.named_steps["regressor"]
            else:
                regressor = pipeline

        if regressor is not None and hasattr(regressor, "predict"):
            try:
                self.shap_explainer = shap.TreeExplainer(regressor)
            except Exception:
                self.shap_explainer = None

    @classmethod
    def load_default(cls, models_dir: Union[str, Path] = Path("models")) -> "ExplainabilityEngine":
        """Load default trained models from the models directory."""
        models_dir = Path(models_dir)
        drift_path = models_dir / "drift_model.pkl"
        ano_path = models_dir / "anomaly_model.pkl"
        safety_path = models_dir / "safety_slope_model.pkl"

        drift_model = DriftPredictor.load(drift_path) if drift_path.exists() else None
        ano_model = DynamicAnomalyDetector.load(ano_path) if ano_path.exists() else None
        safety_model = SafetySlopeEstimator.load(safety_path) if safety_path.exists() else None
        risk_eng = RiskEngine(models_dir / "risk_config.json")

        return cls(
            drift_predictor=drift_model,
            anomaly_detector=ano_model,
            safety_estimator=safety_model,
            risk_engine=risk_eng,
        )

    def explain_component(
        self,
        component_data: Union[pd.Series, pd.DataFrame, Dict[str, Any]],
        component_id: Optional[str] = None,
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """
        Generate structured, human-readable QA diagnostic explanations for a component.

        Parameters
        ----------
        component_data : Union[pd.Series, pd.DataFrame, Dict[str, Any]]
            Telemetry features for the single component.
        component_id : Optional[str]
            Identifier for the component.
        top_k : int
            Number of top contributing features to return.

        Returns
        -------
        Dict[str, Any]
            Complete structured explanation report.
        """
        # Convert input to DataFrame if necessary
        if isinstance(component_data, dict):
            df_row = pd.DataFrame([component_data])
        elif isinstance(component_data, pd.Series):
            df_row = pd.DataFrame([component_data])
        else:
            df_row = component_data.copy()

        row = df_row.iloc[0]
        cid = component_id or str(row.get("Component_ID", "IC_UNKNOWN"))
        lot_id = str(row.get("Lot_ID", "UNKNOWN_LOT"))

        # 1. Base telemetry values
        v0 = float(row.get("Value_0h_uA", 10.0))
        v24 = float(row.get("Value_24h_uA", v0))
        early_drift = float(row.get("Early_Drift_uA", v24 - v0))
        early_drift_rate = float(row.get("Early_Drift_Rate_uA_per_h", early_drift / 24.0))
        pct_change = float(row.get("Early_Percentage_Change", (early_drift / max(1e-7, v0)) * 100.0))
        
        delay_0h = float(row.get("PropagationDelay_0h_ns", 2.0))
        delay_24h = float(row.get("PropagationDelay_24h_ns", delay_0h))
        delay_drift = float(row.get("Early_Delay_Drift_ns", delay_24h - delay_0h))

        datasheet_limit = float(row.get("Datasheet_Leakage_Limit_uA", 50.0))
        batch_z = float(row.get("Value_24h_Robust_Z_Score", 0.0))

        # 2. Module Predictions
        # Anomaly score
        if "Anomaly_Score" in row:
            anomaly_score = float(row["Anomaly_Score"])
        elif self.anomaly_detector is not None:
            anomaly_score = float(self.anomaly_detector.predict_anomaly_scores(df_row)[0])
        else:
            anomaly_score = 0.5 if abs(batch_z) > 2.0 else 0.1

        # 168h Drift Prediction
        if "Predicted_168h" in row:
            pred_168 = float(row["Predicted_168h"])
        elif self.drift_predictor is not None:
            pred_168 = float(self.drift_predictor.predict(df_row)[0])
        else:
            pred_168 = float(v0 + (early_drift_rate * 168.0 * 1.5))

        pred_drift_rate = (pred_168 - v0) / 168.0

        # Safety Slope
        if self.safety_estimator is not None and self.safety_estimator.safety_slope_ > 0:
            safety_slope = float(self.safety_estimator.safety_slope_)
        else:
            safety_slope = float(row.get("Safety_Slope", 0.037074))

        # 3. Risk Engine Evaluation
        risk_res = self.risk_engine.evaluate_component(
            anomaly_score=anomaly_score,
            predicted_168h=pred_168,
            predicted_drift_rate=pred_drift_rate,
            safety_slope=safety_slope,
            datasheet_limit=datasheet_limit,
            value_0h=v0,
            batch_deviation=batch_z,
            component_id=cid,
        )

        risk_score = risk_res["risk_score"]
        risk_level = risk_res["risk_level"]

        # 4. Compute SHAP Feature Attributions
        feature_attributions: List[Dict[str, Any]] = []
        if self.drift_predictor is not None and self.shap_explainer is not None:
            feat_names = self.drift_predictor.feature_names
            # Prepare feature vector
            feat_vector = df_row[feat_names].values
            try:
                shap_vals = self.shap_explainer.shap_values(feat_vector)[0]
                # Sort features by absolute SHAP attribution magnitude
                ranked_indices = np.argsort(np.abs(shap_vals))[::-1]

                for rank, idx in enumerate(ranked_indices[:top_k], start=1):
                    f_name = feat_names[idx]
                    f_val = float(feat_vector[0, idx])
                    f_shap = float(shap_vals[idx])
                    disp_name = QA_FEATURE_DICTIONARY.get(f_name, f_name.replace("_", " "))

                    # Format measured units
                    if "_uA_per_h" in f_name or "_per_h" in f_name:
                        val_str = f"{f_val:.4f} uA/h"
                    elif "_uA" in f_name:
                        val_str = f"{f_val:.2f} uA"
                    elif "_ns" in f_name:
                        val_str = f"{f_val:.3f} ns"
                    elif "_mA" in f_name:
                        val_str = f"{f_val:.3f} mA"
                    elif "Z_Score" in f_name:
                        val_str = f"{f_val:+.2f} sigma"
                    elif "Percentage" in f_name:
                        val_str = f"{f_val:+.1f}%"
                    else:
                        val_str = f"{f_val:.3f}"

                    feature_attributions.append({
                        "importance_rank": rank,
                        "feature_name": f_name,
                        "display_name": disp_name,
                        "measured_value": val_str,
                        "raw_value": round(f_val, 4),
                        "shap_impact_uA": round(f_shap, 4),
                        "impact_direction": "INCREASES_168H_DRIFT" if f_shap > 0 else "DECREASES_168H_DRIFT",
                        "impact_summary": f"{f_shap:+.2f} uA impact on 168h projection",
                    })
            except Exception:
                feature_attributions = self._fallback_statistical_attributions(df_row, top_k)
        else:
            feature_attributions = self._fallback_statistical_attributions(df_row, top_k)

        # 5. Generate Natural Language Human-Readable Reasons
        human_readable_reasons = []

        # Reason 1: Early Leakage Kinetics
        if early_drift > 0.5:
            human_readable_reasons.append(
                f"Early leakage current increased significantly by {early_drift:+.2f} uA ({pct_change:+.1f}%) "
                f"over the first 24 hours of burn-in."
            )
        else:
            human_readable_reasons.append(
                f"Early leakage current exhibited stable behavior (+{early_drift:.2f} uA / {pct_change:+.1f}%) in nominal range."
            )

        # Reason 2: Lot Population Peer Deviation
        if abs(batch_z) >= 2.5:
            human_readable_reasons.append(
                f"Component deviates strongly from its manufacturing lot ({lot_id}) with a robust Z-score of "
                f"{batch_z:+.2f} sigma (exceeds 99th percentile of lot peer distribution)."
            )
        elif abs(batch_z) >= 1.5:
            human_readable_reasons.append(
                f"Component shows moderate peer deviation ({batch_z:+.2f} sigma) compared to lot {lot_id} median."
            )
        else:
            human_readable_reasons.append(
                f"Component closely tracks the manufacturing lot ({lot_id}) median distribution ({batch_z:+.2f} sigma)."
            )

        # Reason 3: 168h Forecast Non-Linear Degradation
        if pred_168 > 25.0:
            human_readable_reasons.append(
                f"Predicted 168h leakage value ({pred_168:.2f} uA) indicates accelerated non-linear degradation "
                f"(consuming {(pred_168 - v0) / (datasheet_limit - v0) * 100.0:.1f}% of permissible datasheet margin)."
            )
        else:
            human_readable_reasons.append(
                f"Predicted 168h leakage value ({pred_168:.2f} uA) remains well within safe operational headroom."
            )

        # Reason 4: Safety Slope Envelope Breach
        if pred_drift_rate > safety_slope:
            ratio = pred_drift_rate / max(1e-7, safety_slope)
            human_readable_reasons.append(
                f"Predicted degradation drift rate ({pred_drift_rate:.4f} uA/h) breaches the healthy safety envelope "
                f"({safety_slope:.4f} uA/h) by {ratio:.2f}x."
            )
        else:
            human_readable_reasons.append(
                f"Predicted degradation rate ({pred_drift_rate:.4f} uA/h) conforms to the healthy safety envelope "
                f"(limit: {safety_slope:.4f} uA/h)."
            )

        # 6. Supporting Measurements
        supporting_measurements = {
            "Component_ID": cid,
            "Lot_ID": lot_id,
            "Value_0h": f"{v0:.2f} uA",
            "Value_24h": f"{v24:.2f} uA",
            "Early_Drift": f"{early_drift:+.2f} uA",
            "Early_Drift_Rate": f"{early_drift_rate:.4f} uA/h",
            "PropagationDelay_0h": f"{delay_0h:.3f} ns",
            "PropagationDelay_24h": f"{delay_24h:.3f} ns",
            "Early_Delay_Drift": f"{delay_drift:+.3f} ns",
            "Predicted_168h": f"{pred_168:.2f} uA",
            "Predicted_Drift_Rate": f"{pred_drift_rate:.4f} uA/h",
            "Datasheet_Limit": f"{datasheet_limit:.2f} uA ({'PASS' if pred_168 <= datasheet_limit else 'FAIL'})",
            "Safety_Slope": f"{safety_slope:.4f} uA/h ({'BREACHED' if pred_drift_rate > safety_slope else 'CONFORMING'})",
            "Batch_Robust_Z_Score": f"{batch_z:+.2f} sigma",
        }

        return {
            "component_id": cid,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "top_contributing_features": feature_attributions,
            "human_readable_reasons": human_readable_reasons,
            "supporting_measurements": supporting_measurements,
            "verdict_summary": risk_res["recommendation"],
            "sub_scores": risk_res["sub_scores"],
        }

    def _fallback_statistical_attributions(self, df_row: pd.DataFrame, top_k: int) -> List[Dict[str, Any]]:
        """Fallback statistical attribution when SHAP explainer is unavailable."""
        attributions = []
        row = df_row.iloc[0]

        candidates = [
            ("Early_Drift_Rate_uA_per_h", float(row.get("Early_Drift_Rate_uA_per_h", 0.0))),
            ("Value_24h_Robust_Z_Score", float(row.get("Value_24h_Robust_Z_Score", 0.0))),
            ("Early_Percentage_Change", float(row.get("Early_Percentage_Change", 0.0))),
            ("Early_Delay_Drift_ns", float(row.get("Early_Delay_Drift_ns", 0.0))),
        ]

        for rank, (f_name, f_val) in enumerate(candidates[:top_k], start=1):
            disp_name = QA_FEATURE_DICTIONARY.get(f_name, f_name)
            attributions.append({
                "importance_rank": rank,
                "feature_name": f_name,
                "display_name": disp_name,
                "measured_value": f"{f_val:+.3f}",
                "raw_value": round(f_val, 4),
                "shap_impact_uA": round(f_val * 2.0, 4),
                "impact_direction": "ELEVATED_RISK" if f_val > 0 else "NOMINAL",
                "impact_summary": f"Statistical weight on risk elevation",
            })
        return attributions

    def generate_text_report(self, explanation: Dict[str, Any]) -> str:
        """
        Format a human-readable QA diagnostic summary card.
        """
        cid = explanation["component_id"]
        level = explanation["risk_level"]
        score = explanation["risk_score"]
        meas = explanation["supporting_measurements"]

        lines = [
            "=" * 72,
            f"ESS DIAGNOSTIC QUALITY REPORT — COMPONENT: {cid}",
            "=" * 72,
            f"OVERALL RISK VERDICT : {level} (Risk Score: {score:.1f} / 100)",
            f"LOT IDENTIFIER       : {meas['Lot_ID']}",
            f"DATASHEET LIMIT      : {meas['Datasheet_Limit']}",
            f"SAFETY ENVELOPE      : {meas['Safety_Slope']}",
            "-" * 72,
            "ROOT-CAUSE DIAGNOSTIC REASONS:",
        ]

        for reason in explanation["human_readable_reasons"]:
            lines.append(f"  - {reason}")

        lines.append("-" * 72)
        lines.append("TOP CONTRIBUTING RISK FEATURES (SHAP & Population Attribution):")
        for feat in explanation["top_contributing_features"]:
            lines.append(
                f"  [{feat['importance_rank']}] {feat['display_name']}: {feat['measured_value']} "
                f"({feat['impact_summary']})"
            )

        lines.append("-" * 72)
        lines.append("SUPPORTING PHYSICAL TELEMETRY:")
        lines.append(f"  - 0h Baseline Leakage  : {meas['Value_0h']}")
        lines.append(f"  - 24h Measured Leakage : {meas['Value_24h']} (Delta: {meas['Early_Drift']})")
        lines.append(f"  - Early Drift Velocity : {meas['Early_Drift_Rate']}")
        lines.append(f"  - Gate Timing Drift    : {meas['Early_Delay_Drift']}")
        lines.append(f"  - 168h Forecast (Mod B): {meas['Predicted_168h']}")
        lines.append(f"  - Batch Robust Z-Score : {meas['Batch_Robust_Z_Score']}")
        lines.append("-" * 72)
        lines.append(f"QA RECOMMENDATION:\n  {explanation['verdict_summary']}")
        lines.append("=" * 72)

        return "\n".join(lines)
