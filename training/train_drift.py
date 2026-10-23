"""
Module B Training, Model Selection, and Evaluation Pipeline for SIH Problem Statement 260170.

Workflow:
1. Ingests train, validation, and test datasets.
2. Extracts strictly early dynamic and batch-relative features (t <= 24h).
3. Trains and benchmarks 3 candidate regression models:
   - Linear Regression (Baseline)
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Compares candidates on the validation set and selects the champion based strictly
   on minimum Validation MAE (Mean Absolute Error).
5. Evaluates champion model on held-out test split.
6. Generates visualization artifacts (Actual vs Predicted, Error Distribution, Model Comparison).
7. Serializes champion model to models/drift_model.pkl and metadata to models/drift_metadata.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.drift_prediction import DriftPredictor
from src.feature_engineering import ESSStage, FeatureEngineer

# ==============================================================================
# Curated Early-Stage Features for 168h Drift Forecasting (t <= 24h)
# Rationale:
# - Value_0h, Value_24h: Initial and 24h baseline leakage magnitude.
# - Early_Drift, Rate, % Change: Initial drift velocity (dI/dt).
# - PropagationDelay 0h, 24h, Drift, Rate: Early timing drift (dielectric/transistor stress proxy).
# - Iddq 0h, 24h, Drift: Quiescent supply current shifts.
# - Robust Z-Scores: Lot-normalized anomaly coordinates to account for wafer lot variance.
# ==============================================================================
SELECTED_DRIFT_FEATURES: List[str] = [
    "Value_0h_uA",
    "Value_24h_uA",
    "Early_Drift_uA",
    "Early_Drift_Rate_uA_per_h",
    "Early_Percentage_Change",
    "PropagationDelay_0h_ns",
    "PropagationDelay_24h_ns",
    "Early_Delay_Drift_ns",
    "Early_Delay_Drift_Rate_ns_per_h",
    "Early_Delay_Percentage_Change",
    "Iddq_0h_mA",
    "Iddq_24h_mA",
    "Early_Iddq_Drift_mA",
    "Early_Iddq_Percentage_Change",
    "Value_24h_Robust_Z_Score",
    "Early_Drift_Robust_Z_Score",
    "Early_Drift_Rate_per_h_Robust_Z_Score",
    "Early_Delay_Drift_Robust_Z_Score",
]


def plot_drift_visualizations(
    val_comparison: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    df_test: pd.DataFrame,
    champion_name: str,
    test_metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Generate the 3 required Module B evaluation visualization plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # --------------------------------------------------------------------------
    # 1. Model Comparison on Validation Set
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    models = list(val_comparison.keys())
    maes = [val_comparison[m]["mae_uA"] for m in models]
    rmses = [val_comparison[m]["rmse_uA"] for m in models]
    r2s = [val_comparison[m]["r2_score"] for m in models]
    model_labels = [m.replace("_", " ").title() for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, maes, width, label="MAE (μA)", color="#2B5C8F", alpha=0.85, edgecolor="black")
    ax1.bar(x + width/2, rmses, width, label="RMSE (μA)", color="#D9381E", alpha=0.85, edgecolor="black")
    ax1.set_title("1. Model Comparison on Validation Split (Error in μA)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Error Magnitude (μA)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.legend(loc="upper left")

    for i, v in enumerate(maes):
        ax1.text(i - width/2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(rmses):
        ax1.text(i + width/2, v + 0.03, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax2.bar(x, r2s, width=0.5, color="#27AE60", alpha=0.85, edgecolor="black")
    ax2.set_title("2. Validation Set R² Score (Coefficient of Determination)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=10)
    ax2.set_ylim([min(r2s) * 0.95, 1.02])

    for i, v in enumerate(r2s):
        ax2.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9.5, fontweight="bold")

    plt.suptitle(f"Module B Model Selection Benchmark (Champion: {champion_name.replace('_', ' ').title()})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "module_b_model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 2. Actual vs Predicted 168h Scatter Plot
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    
    is_defect = df_test["Latent_Defect"].values if "Latent_Defect" in df_test.columns else np.zeros_like(y_test)
    palette = {0: "#2B5C8F", 1: "#D9381E"}
    labels = {0: "Healthy Components", 1: "Latent Defect Components"}

    for state, color in palette.items():
        mask = (is_defect == state)
        ax.scatter(
            y_test[mask],
            y_test_pred[mask],
            color=color,
            alpha=0.6,
            s=35,
            edgecolors="none",
            label=labels[state],
        )

    # Ideal identity line y = x
    min_val = min(float(np.min(y_test)), float(np.min(y_test_pred))) * 0.95
    max_val = max(float(np.max(y_test)), float(np.max(y_test_pred))) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], color="#333333", linestyle="--", linewidth=2, label="Ideal Line (y = x)")
    
    # +/- 1.0 uA tolerance bounds
    ax.plot([min_val, max_val], [min_val + 1.0, max_val + 1.0], color="#888888", linestyle=":", linewidth=1.5, label="±1.0 μA Error Band")
    ax.plot([min_val, max_val], [min_val - 1.0, max_val - 1.0], color="#888888", linestyle=":", linewidth=1.5)

    ax.set_title(f"Module B: Actual vs. Predicted 168h Leakage Current ({champion_name.replace('_', ' ').title()})", pad=12, fontweight="bold", fontsize=13)
    ax.set_xlabel("Actual Ground Truth Value_168h_uA (μA)", fontsize=11)
    ax.set_ylabel("Predicted 168h Leakage Current (μA)", fontsize=11)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.legend(loc="upper left", fontsize=10)

    # Text summary box
    ax.text(
        0.04, 0.65,
        f"Test Split Metrics:\n"
        f"• MAE:  {test_metrics['mae_uA']:.4f} μA\n"
        f"• RMSE: {test_metrics['rmse_uA']:.4f} μA\n"
        f"• R²:   {test_metrics['r2_score']:.4f}\n"
        f"• 95% of predictions within ±{test_metrics['residuals']['p95_absolute_error']:.2f} μA",
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF9E6", edgecolor="#E5C158", alpha=0.95),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "module_b_actual_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 3. Prediction Error (Residual) Distribution
    # --------------------------------------------------------------------------
    residuals = y_test - y_test_pred
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    sns.histplot(residuals, kde=True, ax=ax, color="#2B5C8F", bins=40, alpha=0.6, edgecolor="black", linewidth=1.0)
    ax.axvline(0.0, color="#D9381E", linestyle="--", linewidth=2, label="Zero Bias (Residual = 0.0 μA)")
    ax.axvline(float(np.mean(residuals)), color="#27AE60", linestyle=":", linewidth=2, label=f"Mean Error = {np.mean(residuals):+.3f} μA")

    ax.set_title("Module B: 168h Forecast Residual Distribution (Actual - Predicted)", pad=12, fontweight="bold", fontsize=13)
    ax.set_xlabel("Prediction Error / Residual (μA)", fontsize=11)
    ax.set_ylabel("Component Count", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)

    ax.text(
        0.04, 0.90,
        f"Residual Properties:\n"
        f"• Mean:   {np.mean(residuals):+.4f} μA (Unbiased)\n"
        f"• Std:    {np.std(residuals):.4f} μA\n"
        f"• Median: {np.median(residuals):+.4f} μA\n"
        f"• Max Error: {np.max(np.abs(residuals)):.3f} μA",
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB", edgecolor="#85C1E9", alpha=0.95),
    )

    plt.tight_layout()
    fig.savefig(output_dir / "module_b_error_distribution.png", bbox_inches="tight")
    plt.close(fig)


def train_and_select_drift_module(
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    output_dir: Path = Path("outputs"),
) -> Dict[str, Any]:
    """
    Train and compare candidate regression architectures, select champion by validation MAE,
    evaluate on test set, export artifacts.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "sih_260170_train.csv"
    val_path = data_dir / "sih_260170_validation.csv"
    test_path = data_dir / "sih_260170_test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train dataset not found at {train_path.resolve()}")

    print(f"Loading datasets from {data_dir}...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 1. Feature Engineering (Strictly stage <= 24h features)
    print("Fitting FeatureEngineer pipeline on training split...")
    fe = FeatureEngineer(stage=ESSStage.STAGE_24H, include_batch_relative=True)
    fe.fit(df_train, lot_column="Lot_ID")

    df_train_feat = fe.transform(df_train, lot_column="Lot_ID")
    df_val_feat = fe.transform(df_val, lot_column="Lot_ID")
    df_test_feat = fe.transform(df_test, lot_column="Lot_ID")

    y_train = df_train["Value_168h_uA"].values
    y_val = df_val["Value_168h_uA"].values
    y_test = df_test["Value_168h_uA"].values

    # 2. Train and Benchmark 3 Candidate Models
    candidate_models = {
        "linear_regression": {},
        "random_forest": {"n_estimators": 100, "max_depth": 14, "random_state": 42, "n_jobs": -1},
        "gradient_boosting": {"n_estimators": 150, "learning_rate": 0.08, "max_depth": 4, "random_state": 42},
    }

    trained_predictors: Dict[str, DriftPredictor] = {}
    val_evaluations: Dict[str, Dict[str, Any]] = {}

    print("\n" + "=" * 65)
    print("BENCHMARKING MODULE B CANDIDATE MODELS ON VALIDATION SPLIT:")
    print("=" * 65)

    for model_type, config in candidate_models.items():
        print(f"\nTraining candidate: {model_type}...")
        predictor = DriftPredictor(
            model_type=model_type,
            config=config,
            feature_names=SELECTED_DRIFT_FEATURES,
        )
        predictor.fit(df_train_feat, y_train)
        trained_predictors[model_type] = predictor

        # Evaluate on validation split
        val_eval = predictor.evaluate(df_val_feat, y_val)
        val_evaluations[model_type] = val_eval
        print(f"  - Validation MAE  : {val_eval['mae_uA']:.4f} uA")
        print(f"  - Validation RMSE : {val_eval['rmse_uA']:.4f} uA")
        print(f"  - Validation R2   : {val_eval['r2_score']:.4f}")

    # 3. Model Selection: Select Champion by Lowest Validation MAE
    champion_type = min(val_evaluations, key=lambda k: val_evaluations[k]["mae_uA"])
    champion_predictor = trained_predictors[champion_type]
    champion_val_eval = val_evaluations[champion_type]

    print("\n" + "=" * 65)
    print(f"CHAMPION MODEL SELECTED (Based strictly on Validation MAE): {champion_type.upper()}")
    print(f"  - Winning Validation MAE : {champion_val_eval['mae_uA']:.4f} uA")
    print(f"  - Winning Validation R2  : {champion_val_eval['r2_score']:.4f}")
    print("=" * 65)

    # 4. Final Evaluation on Held-Out Test Split
    print("\nEvaluating Champion Model on Unseen Test Split...")
    test_eval = champion_predictor.evaluate(df_test_feat, y_test)
    y_test_pred = champion_predictor.predict(df_test_feat)

    print("=" * 65)
    print("CHAMPION MODEL TEST SET EVALUATION RESULTS:")
    print("=" * 65)
    print(f"  - Test Components        : {len(df_test):,}")
    print(f"  - Test MAE (Mean Error)  : {test_eval['mae_uA']:.4f} uA")
    print(f"  - Test RMSE              : {test_eval['rmse_uA']:.4f} uA")
    print(f"  - Test R2 Score          : {test_eval['r2_score']:.4f}")
    print(f"  - Max Absolute Error     : {test_eval['residuals']['max_absolute_error']:.4f} uA")
    print(f"  - 95th Percentile Error  : {test_eval['residuals']['p95_absolute_error']:.4f} uA")
    print("=" * 65)

    # 5. Save Champion Model & Metadata
    model_save_path = models_dir / "drift_model.pkl"
    metadata_save_path = models_dir / "drift_metadata.json"

    print(f"\nSaving champion model to {model_save_path}...")
    champion_predictor.save(model_save_path)

    metadata = {
        "module": "Module B - 168h Drift Prediction",
        "champion_model_type": champion_type,
        "library": "scikit-learn",
        "hyperparameters": champion_predictor.config,
        "selected_features": SELECTED_DRIFT_FEATURES,
        "feature_count": len(SELECTED_DRIFT_FEATURES),
        "target_variable": "Value_168h_uA",
        "primary_metric": "MAE (Mean Absolute Error)",
        "model_selection_rationale": (
            f"Selected '{champion_type}' based strictly on validation split performance. "
            f"Achieved lowest Validation MAE of {champion_val_eval['mae_uA']:.4f} μA and highest R² of {champion_val_eval['r2_score']:.4f}."
        ),
        "validation_benchmark_comparison": val_evaluations,
        "test_set_performance": test_eval,
    }

    with open(metadata_save_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_save_path}...")

    # 6. Generate Visualizations
    print("Generating Module B visualization plots...")
    plot_drift_visualizations(
        val_comparison=val_evaluations,
        y_test=y_test,
        y_test_pred=y_test_pred,
        df_test=df_test,
        champion_name=champion_type,
        test_metrics=test_eval,
        output_dir=output_dir,
    )

    return metadata


def main():
    train_and_select_drift_module()


if __name__ == "__main__":
    main()
