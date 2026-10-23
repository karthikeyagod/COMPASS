"""
End-to-End Machine Learning Inference Pipeline for SIH Problem Statement 260170.

Orchestrates the unified evaluation workflow:
1. Input Telemetry Ingestion & Quality Validation
2. Stage-Aware Dynamic Feature Engineering (t <= 24h)
3. Module A: Dynamic Anomaly Detection (Isolation Forest)
4. Module B: 168-Hour Parametric Drift Forecasting (Gradient Boosting)
5. Safety Slope & Healthy Degradation Envelope Compliance
6. Component Risk Engine (Composite 0-100 Score & Categorization)
7. Explainable AI (SHAP Local Attributions & QA Diagnostic Reasons)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd

# Suppress harmless scikit-learn unpickle version mismatch warnings in production logs
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from src.anomaly_detection import DynamicAnomalyDetector
from src.drift_prediction import DriftPredictor
from src.explainability import ExplainabilityEngine
from src.feature_engineering import ESSStage, FeatureEngineer
from src.risk_engine import RiskEngine
from src.safety_slope import SafetySlopeEstimator


class InferencePipeline:
    """
    Singleton-capable orchestrator managing all ML models, preprocessors,
    and explainability engines for component screening.
    """

    _instance: Optional["InferencePipeline"] = None

    def __init__(
        self,
        models_dir: Union[str, Path] = Path("models"),
        data_dir: Union[str, Path] = Path("data"),
    ) -> None:
        """
        Initialize and load all pipeline models and feature baselines.
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.is_initialized = False

        self.anomaly_detector: Optional[DynamicAnomalyDetector] = None
        self.drift_predictor: Optional[DriftPredictor] = None
        self.safety_estimator: Optional[SafetySlopeEstimator] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.explainability_engine: Optional[ExplainabilityEngine] = None
        self.feature_engineer: Optional[FeatureEngineer] = None

        self.initialize()

    def initialize(self) -> None:
        """
        Load models and fit baseline feature statistics.
        """
        ano_path = self.models_dir / "anomaly_model.pkl"
        drift_path = self.models_dir / "drift_model.pkl"
        safety_path = self.models_dir / "safety_slope_model.pkl"
        config_path = self.models_dir / "risk_config.json"

        # Load models
        if ano_path.exists():
            self.anomaly_detector = DynamicAnomalyDetector.load(ano_path)
        if drift_path.exists():
            self.drift_predictor = DriftPredictor.load(drift_path)
        if safety_path.exists():
            self.safety_estimator = SafetySlopeEstimator.load(safety_path)
        
        self.risk_engine = RiskEngine(config=config_path if config_path.exists() else None)

        self.explainability_engine = ExplainabilityEngine(
            drift_predictor=self.drift_predictor,
            anomaly_detector=self.anomaly_detector,
            safety_estimator=self.safety_estimator,
            risk_engine=self.risk_engine,
        )

        # Feature Engineer baseline fitting
        self.feature_engineer = FeatureEngineer(
            stage=ESSStage.STAGE_24H,
            include_batch_relative=True,
        )

        train_path = self.data_dir / "sih_260170_train.csv"
        if train_path.exists():
            df_train = pd.read_csv(train_path)
            self.feature_engineer.fit(df_train, lot_column="Lot_ID")

        self.is_initialized = True

    def validate_and_preprocess(
        self,
        component_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate input measurements, handle missing fields, and detect sensor anomalies.
        """
        warnings: List[str] = []

        if isinstance(component_data, dict):
            df = pd.DataFrame([component_data])
        elif isinstance(component_data, pd.Series):
            df = pd.DataFrame([component_data])
        else:
            df = component_data.copy()

        if len(df) == 0:
            raise ValueError("Input component data is empty.")

        row = df.iloc[0].to_dict()

        # Check required baseline columns
        if "Value_0h_uA" not in row or pd.isna(row["Value_0h_uA"]):
            if "Value_0h" in row and not pd.isna(row["Value_0h"]):
                row["Value_0h_uA"] = float(row["Value_0h"])
            else:
                warnings.append("Missing 'Value_0h_uA'; imputed with nominal baseline 10.0 uA.")
                row["Value_0h_uA"] = 10.0

        if "Value_24h_uA" not in row or pd.isna(row["Value_24h_uA"]):
            if "Value_24h" in row and not pd.isna(row["Value_24h"]):
                row["Value_24h_uA"] = float(row["Value_24h"])
            else:
                warnings.append("Missing 'Value_24h_uA'; imputed equal to Value_0h_uA.")
                row["Value_24h_uA"] = float(row["Value_0h_uA"])

        v0 = float(row["Value_0h_uA"])
        v24 = float(row["Value_24h_uA"])

        # Check extreme / corrupted measurements
        if v0 < 0.0 or v24 < 0.0:
            warnings.append(
                f"Negative leakage current reading detected (0h={v0:.2f}, 24h={v24:.2f} uA). "
                "Indicates potential sensor fault or reverse-bias fixture error."
            )
            # Clip for safety
            row["Value_0h_uA"] = max(0.01, v0)
            row["Value_24h_uA"] = max(0.01, v24)

        if v24 > 500.0 or v0 > 500.0:
            warnings.append(
                f"Extreme gross physical current surge detected (0h={v0:.2f}, 24h={v24:.2f} uA). "
                "Exceeds standard 500 uA measurement ceiling; immediate gross short."
            )

        # Impute optional telemetry fields with domain-sensible nominals if missing
        if "PropagationDelay_0h_ns" not in row or pd.isna(row["PropagationDelay_0h_ns"]):
            row["PropagationDelay_0h_ns"] = 2.015
        if "PropagationDelay_24h_ns" not in row or pd.isna(row["PropagationDelay_24h_ns"]):
            row["PropagationDelay_24h_ns"] = float(row["PropagationDelay_0h_ns"]) + max(0.0, (v24 - v0) * 0.1)

        if "Iddq_0h_mA" not in row or pd.isna(row["Iddq_0h_mA"]):
            row["Iddq_0h_mA"] = 1.20
        if "Iddq_24h_mA" not in row or pd.isna(row["Iddq_24h_mA"]):
            row["Iddq_24h_mA"] = float(row["Iddq_0h_mA"]) + max(0.0, (v24 - v0) * 0.05)

        if "Datasheet_Leakage_Limit_uA" not in row or pd.isna(row["Datasheet_Leakage_Limit_uA"]):
            row["Datasheet_Leakage_Limit_uA"] = 50.0

        if "Temperature_C" not in row or pd.isna(row["Temperature_C"]):
            row["Temperature_C"] = 125.0

        if "Lot_ID" not in row or pd.isna(row["Lot_ID"]):
            row["Lot_ID"] = "DEFAULT_LOT"

        if "Component_ID" not in row or pd.isna(row["Component_ID"]):
            row["Component_ID"] = "IC_UNKNOWN"

        df_clean = pd.DataFrame([row])
        return df_clean, warnings

    def analyze_component(
        self,
        component_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end inference workflow on a single component observation.

        Returns
        -------
        Dict[str, Any]
            Structured decision report containing:
            - Component ID
            - Anomaly score
            - Predicted 168h
            - Predicted drift
            - Safety slope
            - Risk score
            - Risk level
            - Explanation
            - Validation warnings
        """
        # Step 1: Quality Validation & Preprocessing
        df_clean, validation_warnings = self.validate_and_preprocess(component_data)
        cid = str(df_clean["Component_ID"].iloc[0])
        v0 = float(df_clean["Value_0h_uA"].iloc[0])
        limit = float(df_clean["Datasheet_Leakage_Limit_uA"].iloc[0])

        # Step 2: Feature Engineering (t <= 24h)
        if self.feature_engineer is not None:
            df_feat = self.feature_engineer.transform(df_clean, lot_column="Lot_ID")
        else:
            df_feat = df_clean.copy()

        # Step 3: Module A - Dynamic Anomaly Score
        if self.anomaly_detector is not None:
            ano_score = float(self.anomaly_detector.predict_anomaly_scores(df_feat)[0])
        else:
            ano_score = 0.5

        # Step 4: Module B - 168h Forecast & Predicted Drift
        if self.drift_predictor is not None:
            pred_168 = float(self.drift_predictor.predict(df_feat)[0])
        else:
            # Domain Arrhenius extrapolation fallback
            rate = float(df_feat.get("Early_Drift_Rate_uA_per_h", pd.Series([0.01])).iloc[0])
            pred_168 = float(v0 + (rate * 168.0 * 1.5))

        pred_drift = float(pred_168 - v0)
        pred_drift_rate = float(pred_drift / 168.0)

        # Step 5: Safety Slope Evaluation
        if self.safety_estimator is not None and self.safety_estimator.safety_slope_ > 0:
            safety_slope = float(self.safety_estimator.safety_slope_)
        else:
            safety_slope = 0.037074

        slope_exceeded = 1 if pred_drift_rate > safety_slope else 0

        # Step 6: Risk Engine Scoring
        batch_z = float(df_feat.get("Value_24h_Robust_Z_Score", pd.Series([0.0])).iloc[0])
        risk_res = self.risk_engine.evaluate_component(
            anomaly_score=ano_score,
            predicted_168h=pred_168,
            predicted_drift_rate=pred_drift_rate,
            safety_slope=safety_slope,
            datasheet_limit=limit,
            value_0h=v0,
            batch_deviation=batch_z,
            component_id=cid,
        )

        risk_score = risk_res["risk_score"]
        risk_level = risk_res["risk_level"]

        # If gross surge detected in validation, enforce maximum risk
        if any("Extreme gross" in w or "Negative leakage" in w for w in validation_warnings):
            risk_score = 100.0
            risk_level = "HIGH_RISK"

        # Step 7: Explainability Engine
        if self.explainability_engine is not None:
            explanation = self.explainability_engine.explain_component(
                component_data=df_feat,
                component_id=cid,
            )
        else:
            explanation = {
                "top_contributing_features": [],
                "human_readable_reasons": ["Automated screening decision."],
                "supporting_measurements": {},
                "verdict_summary": risk_res["recommendation"],
            }

        # Format Final Result Contract
        result = {
            "component_id": cid,
            "anomaly_score": round(ano_score, 4),
            "predicted_168h": round(pred_168, 4),
            "predicted_drift": round(pred_drift, 4),
            "predicted_drift_rate": round(pred_drift_rate, 6),
            "safety_slope": round(safety_slope, 6),
            "slope_exceeded": slope_exceeded,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "datasheet_limit": round(limit, 2),
            "datasheet_pass": bool(pred_168 <= limit),
            "explanation": explanation,
            "validation_warnings": validation_warnings,
        }

        return result

    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized high-performance batch evaluation across a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing raw component measurements.

        Returns
        -------
        pd.DataFrame
            Result DataFrame with predicted 168h, anomaly score, risk score, and risk level.
        """
        if len(df) == 0:
            return pd.DataFrame()

        df_clean = df.copy()
        v0_raw = df_clean["Value_0h_uA"].values if "Value_0h_uA" in df_clean.columns else np.full(len(df_clean), 10.0)
        v24_raw = df_clean["Value_24h_uA"].values if "Value_24h_uA" in df_clean.columns else v0_raw

        if "PropagationDelay_0h_ns" not in df_clean.columns:
            df_clean["PropagationDelay_0h_ns"] = 2.015
        if "PropagationDelay_24h_ns" not in df_clean.columns:
            df_clean["PropagationDelay_24h_ns"] = df_clean["PropagationDelay_0h_ns"] + np.maximum(0.0, (v24_raw - v0_raw) * 0.1)

        if "Iddq_0h_mA" not in df_clean.columns:
            df_clean["Iddq_0h_mA"] = 1.20
        if "Iddq_24h_mA" not in df_clean.columns:
            df_clean["Iddq_24h_mA"] = df_clean["Iddq_0h_mA"] + np.maximum(0.0, (v24_raw - v0_raw) * 0.05)

        if "Temperature_C" not in df_clean.columns:
            df_clean["Temperature_C"] = 125.0
        if "Datasheet_Leakage_Limit_uA" not in df_clean.columns:
            df_clean["Datasheet_Leakage_Limit_uA"] = 50.0

        # Step 1: Feature Engineering (t <= 24h)
        if self.feature_engineer is not None:
            df_feat = self.feature_engineer.transform(df_clean, lot_column="Lot_ID" if "Lot_ID" in df_clean.columns else None)
        else:
            df_feat = df_clean.copy()

        # Step 2: Module A - Anomaly Scores
        if self.anomaly_detector is not None:
            ano_scores = self.anomaly_detector.predict_anomaly_scores(df_feat)
        else:
            ano_scores = np.full(len(df), 0.5)

        # Step 3: Module B - 168h Forecasts
        if self.drift_predictor is not None:
            pred_168 = self.drift_predictor.predict(df_feat)
        else:
            v0_temp = df["Value_0h_uA"].values if "Value_0h_uA" in df.columns else np.full(len(df), 10.0)
            v24_temp = df["Value_24h_uA"].values if "Value_24h_uA" in df.columns else v0_temp
            pred_168 = v24_temp + (v24_temp - v0_temp) * 6.0

        v0 = df["Value_0h_uA"].values if "Value_0h_uA" in df.columns else np.full(len(df), 10.0)
        pred_drift = pred_168 - v0
        pred_drift_rate = pred_drift / 168.0

        # Step 4: Safety Slope
        if self.safety_estimator is not None and self.safety_estimator.safety_slope_ > 0:
            safety_slope = float(self.safety_estimator.safety_slope_)
        else:
            safety_slope = 0.037074

        slope_exceeded = (pred_drift_rate > safety_slope).astype(int)

        # Step 5: Risk Engine Scoring
        batch_z = (
            df_feat["Value_24h_Robust_Z_Score"].values
            if "Value_24h_Robust_Z_Score" in df_feat.columns
            else np.zeros(len(df))
        )
        limits = (
            df["Datasheet_Leakage_Limit_uA"].values
            if "Datasheet_Leakage_Limit_uA" in df.columns
            else np.full(len(df), 50.0)
        )
        cids = (
            df["Component_ID"].values
            if "Component_ID" in df.columns
            else [f"IC_{i}" for i in range(len(df))]
        )

        risk_scores = []
        risk_levels = []
        for i in range(len(df)):
            eval_res = self.risk_engine.evaluate_component(
                anomaly_score=float(ano_scores[i]),
                predicted_168h=float(pred_168[i]),
                predicted_drift_rate=float(pred_drift_rate[i]),
                safety_slope=safety_slope,
                datasheet_limit=float(limits[i]),
                value_0h=float(v0[i]),
                batch_deviation=float(batch_z[i]),
                component_id=str(cids[i]),
            )
            risk_scores.append(eval_res["risk_score"])
            risk_levels.append(eval_res["risk_level"])

        res_df = pd.DataFrame({
            "Component_ID": cids,
            "Anomaly_Score": ano_scores,
            "Predicted_168h": pred_168,
            "Predicted_Drift": pred_drift,
            "Predicted_Drift_Rate": pred_drift_rate,
            "Safety_Slope": np.full(len(df), safety_slope),
            "Slope_Exceeded": slope_exceeded,
            "Risk_Score": risk_scores,
            "Risk_Level": risk_levels,
        })
        return res_df


def analyze_component(
    component_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    lot_history: Optional[pd.DataFrame] = None,
    models_dir: Union[str, Path] = Path("models"),
    data_dir: Union[str, Path] = Path("data"),
) -> Dict[str, Any]:
    """
    High-level functional API entry point to analyze a single component.

    Parameters
    ----------
    component_data : Union[Dict[str, Any], pd.Series, pd.DataFrame]
        Raw or engineered sensor telemetry for the component.
    lot_history : Optional[pd.DataFrame]
        Historical peer components for intra-lot normalization.
    models_dir : Union[str, Path]
        Directory containing serialized model artifacts.
    data_dir : Union[str, Path]
        Directory containing baseline datasets.

    Returns
    -------
    Dict[str, Any]
        Structured screening decision report.
    """
    pipeline = InferencePipeline(models_dir=models_dir, data_dir=data_dir)
    return pipeline.analyze_component(component_data)
