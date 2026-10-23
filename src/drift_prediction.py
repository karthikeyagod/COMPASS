"""
Drift Prediction Module (Module B) for SIH Problem Statement 260170.

Module B forecasts the final 168-hour Environmental Stress Screening (ESS) degradation
state (Value_168h_uA) using exclusively early burn-in measurements (t <= 24h).

Theoretical Foundation:
- Inputs: Initial baseline parameters at t=0h (Value_0h, PropagationDelay_0h, Iddq_0h),
  early kinetic drift at t=24h (Value_24h, Early_Drift, Early_Drift_Rate, Early_Percentage_Change,
  Early_Delay_Drift), and peer-relative lot z-scores.
- Target: Final leakage current at t=168h (Value_168h_uA).
- Physical Mechanism: Arrhenius accelerated thermal kinetics causes exponential/power-law
  parametric drift in sub-surface dielectric defects. Early drift rates (dI/dt at 24h)
  serve as strong mathematical determinants of the cumulative 168h degradation state.
- Evaluation Metric: Mean Absolute Error (MAE):
      MAE = (1 / N) * sum(|y_i - y_hat_i|)
  Directly represents the physical deviation in microamperes (uA) between predicted and
  actual 168-hour leakage current.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import (
    DataLeakageError,
    ESSStage,
    validate_feature_availability,
)


class DriftPredictor:
    """
    Module B: 168-Hour Parametric Drift Forecasting Engine.
    
    Predicts Value_168h_uA using early-stage screening features (t <= 24h).
    Supports Linear Regression, Random Forest, and Gradient Boosting backends.
    """

    SUPPORTED_MODELS = ["linear_regression", "random_forest", "gradient_boosting"]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        config: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_type : str
            Model architecture ('linear_regression', 'random_forest', 'gradient_boosting').
        config : Optional[Dict[str, Any]]
            Hyperparameters for the chosen regressor.
        feature_names : Optional[List[str]]
            Curated list of early feature names (t <= 24h).
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose from {self.SUPPORTED_MODELS}")

        self.model_type = model_type
        self.config = config or {}
        self.feature_names: List[str] = feature_names or []

        if self.feature_names:
            self._validate_feature_names(self.feature_names)

        self.model: Optional[Union[Pipeline, RandomForestRegressor, GradientBoostingRegressor]] = None

    def _validate_feature_names(self, features: List[str]) -> None:
        """Enforce zero lookahead leakage rules on input feature names."""
        disallowed = {"Value_168h_uA", "Value_96h_uA", "Total_Drift_uA", "Latent_Defect", "Absolute_Limit_Pass"}
        for f in features:
            if f in disallowed:
                raise DataLeakageError(
                    f"Data Leakage Violation: Feature '{f}' is unavailable at stage <= 24h and cannot be used in Module B."
                )
        validate_feature_availability(features, current_stage=ESSStage.STAGE_24H)

    def _init_underlying_model(self) -> Any:
        """Initialize regressor based on model_type and config."""
        from sklearn.impute import SimpleImputer

        if self.model_type == "linear_regression":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression(**self.config)),
            ])
        elif self.model_type == "random_forest":
            default_rf_config = {
                "n_estimators": 100,
                "max_depth": 12,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_rf_config.update(self.config)
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", RandomForestRegressor(**default_rf_config)),
            ])
        elif self.model_type == "gradient_boosting":
            default_gb_config = {
                "n_estimators": 150,
                "learning_rate": 0.08,
                "max_depth": 4,
                "random_state": 42,
            }
            default_gb_config.update(self.config)
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("regressor", GradientBoostingRegressor(**default_gb_config)),
            ])

    def fit(self, df: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "DriftPredictor":
        """
        Fit the drift prediction model on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered DataFrame containing early features.
        y : Union[pd.Series, np.ndarray]
            Target continuous 168h values (Value_168h_uA).

        Returns
        -------
        DriftPredictor
            Fitted predictor instance.
        """
        if not self.feature_names:
            raise ValueError("feature_names list must be provided before fitting.")

        self._validate_feature_names(self.feature_names)

        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

        X = df[self.feature_names].values
        y_arr = np.asarray(y, dtype=float)

        self.model = self._init_underlying_model()
        self.model.fit(X, y_arr)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict Value_168h_uA for given early screening observations.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered DataFrame containing early features.

        Returns
        -------
        np.ndarray
            Predicted continuous 168-hour leakage values in uA.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Input DataFrame is missing required features: {missing}")

        X = df[self.feature_names].values
        raw_preds = self.model.predict(X)

        # Enforce physical degradation monotonicity: 168h continuous stress leakage cannot be less than 24h reading
        if "Value_24h_uA" in df.columns:
            v24 = df["Value_24h_uA"].values
            if "Early_Drift_uA" in df.columns:
                early_drift = np.maximum(0.0, df["Early_Drift_uA"].values)
                linear_extrap = v24 + (early_drift * 6.0)
                # If component is in extreme out-of-domain drift (e.g. v24 > 25 uA), prevent tree saturation
                return np.where(v24 > 25.0, np.maximum(raw_preds, linear_extrap), np.maximum(raw_preds, v24))
            return np.maximum(raw_preds, v24)

        return raw_preds

    def evaluate(self, df: pd.DataFrame, y_true: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Compute comprehensive regression evaluation metrics: MAE, RMSE, R2, and residuals.

        Parameters
        ----------
        df : pd.DataFrame
            Input feature DataFrame.
        y_true : Union[pd.Series, np.ndarray]
            Ground truth 168h continuous values.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing MAE, RMSE, R2, and residual statistics.
        """
        y_arr = np.asarray(y_true, dtype=float)
        y_pred = self.predict(df)
        residuals = y_arr - y_pred

        mae = float(mean_absolute_error(y_arr, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_arr, y_pred)))
        r2 = float(r2_score(y_arr, y_pred))

        return {
            "model_type": self.model_type,
            "mae_uA": round(mae, 6),
            "rmse_uA": round(rmse, 6),
            "r2_score": round(r2, 6),
            "residuals": {
                "mean_error": round(float(np.mean(residuals)), 6),
                "std_error": round(float(np.std(residuals)), 6),
                "median_error": round(float(np.median(residuals)), 6),
                "max_absolute_error": round(float(np.max(np.abs(residuals))), 6),
                "p95_absolute_error": round(float(np.percentile(np.abs(residuals), 95)), 6),
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize trained model and metadata to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "config": self.config,
            "feature_names": self.feature_names,
            "model": self.model,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DriftPredictor":
        """Deserialize trained model from disk."""
        payload = joblib.load(filepath)
        instance = cls(
            model_type=payload["model_type"],
            config=payload["config"],
            feature_names=payload["feature_names"],
        )
        instance.model = payload["model"]
        return instance
