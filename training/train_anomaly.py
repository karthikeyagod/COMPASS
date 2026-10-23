"""
Module A Training and Evaluation Pipeline for SIH Problem Statement 260170.

Workflow:
1. Load train, validation, and test datasets.
2. Extract strictly early-stage dynamic and lot-relative features (t <= 24h).
3. Benchmark statistical baseline detectors (Z-score, Robust Z-score, IQR).
4. Train DynamicAnomalyDetector (Isolation Forest) on training split.
5. Calibrate decision threshold on validation split prioritizing high Recall (>= 98%).
6. Evaluate test split metrics (Recall, Precision, F1, PR-AUC, ROC-AUC, Escapes).
7. Generate visualization plots (Confusion Matrix, PR Curve, Score Distribution).
8. Save trained model to models/anomaly_model.pkl and metadata to models/anomaly_metadata.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve

from src.anomaly_detection import (
    DynamicAnomalyDetector,
    IQRDetector,
    RobustZScoreDetector,
    ZScoreDetector,
)
from src.feature_engineering import ESSStage, FeatureEngineer

# ==============================================================================
# Curated Early-Stage Feature Selection (t <= 24h)
# Rationale:
# - Kinetic Drift (Value_0h, Value_24h, Early_Drift, Rate, % Change): Captures early leakage acceleration.
# - Dynamic Delay Drift (Delay 0h, 24h, Drift, Rate, % Change): Captures timing degradation.
# - Dynamic Iddq Drift (Iddq 0h, 24h, Drift, % Change): Captures switching current shifts.
# - Peer-Relative Robust Z-Scores: Normalizes component kinetics against lot-level variance.
# ==============================================================================
SELECTED_EARLY_FEATURES: List[str] = [
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


def plot_module_a_visualizations(
    y_test: np.ndarray,
    scores_test: np.ndarray,
    y_pred_test: np.ndarray,
    threshold: float,
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Generate the 3 required Module A evaluation visualization artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # --------------------------------------------------------------------------
    # 1. Confusion Matrix Plot
    # --------------------------------------------------------------------------
    tp = metrics["confusion_matrix"]["tp"]
    tn = metrics["confusion_matrix"]["tn"]
    fp = metrics["confusion_matrix"]["fp"]
    fn = metrics["confusion_matrix"]["fn"]

    cm_matrix = np.array([[tn, fp], [fn, tp]])
    annot = np.array([
        [f"True Negatives\n(Healthy Pass)\n\n{tn:,}", f"False Positives\n(False Alarms)\n\n{fp:,}"],
        [f"False Negatives\n(SILENT ESCAPES)\n\n{fn:,} ({metrics['false_negative_rate_fnr']:.1%})",
         f"True Positives\n(CAUGHT DEFECTS)\n\n{tp:,} ({metrics['recall']:.1%})"],
    ])

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.heatmap(
        cm_matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=2,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title("Module A: Dynamic Anomaly Detection Confusion Matrix\n(Isolation Forest on 24h Early Drift Features)", pad=16, fontsize=13, fontweight="bold")
    ax.set_xlabel("Module A Prediction Verdict\n(0 = Nominal / Pass, 1 = Anomaly / Flagged)", labelpad=10, fontsize=11)
    ax.set_ylabel("True Ground Truth (Latent_Defect)\n(0 = Healthy, 1 = Defective)", labelpad=10, fontsize=11)
    ax.set_xticklabels(["Screening: PASS (0)", "Screening: FLAG (1)"], fontsize=10)
    ax.set_yticklabels(["Actual: Healthy (0)", "Actual: Defective (1)"], fontsize=10, rotation=0)

    plt.figtext(
        0.5, -0.05,
        f"Module A Test Performance (vs Traditional Baseline):\n"
        f"• Recall = {metrics['recall']:.1%} (Baseline was 0.0%) | "
        f"• Precision = {metrics['precision']:.1%} | "
        f"• F1 = {metrics['f1_score']:.1%}\n"
        f"• Escapes Eliminated: Caught {tp:,} / {tp+fn:,} defective ICs within 24h!",
        ha="center",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#27AE60", alpha=0.95),
    )
    plt.tight_layout()
    fig.savefig(output_dir / "module_a_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 2. Precision-Recall Curve Plot
    # --------------------------------------------------------------------------
    precisions, recalls, thresholds = precision_recall_curve(y_test, scores_test)
    
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    ax.plot(recalls, precisions, color="#2B5C8F", linewidth=2.5, label=f"Module A PR Curve (PR-AUC = {metrics['pr_auc']:.4f})")
    
    # Highlight the calibrated operating point
    ax.scatter(
        [metrics["recall"]],
        [metrics["precision"]],
        color="#D9381E",
        s=120,
        zorder=5,
        label=f"Calibrated Operating Threshold ({threshold:.3f})\nRecall = {metrics['recall']:.1%}, Prec = {metrics['precision']:.1%}",
    )

    ax.set_title("Module A: Precision-Recall Curve for Latent Defect Detection", pad=12, fontweight="bold", fontsize=13)
    ax.set_xlabel("Recall (Defect Detection Rate)", fontsize=11)
    ax.set_ylabel("Precision (True Defect Proportion)", fontsize=11)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "module_a_pr_curve.png", bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------------------------------
    # 3. Anomaly Score Distribution Plot
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    sns.kdeplot(scores_test[y_test == 0], ax=ax, color="#2B5C8F", fill=True, alpha=0.35, label="Healthy Components (0)", linewidth=2)
    sns.kdeplot(scores_test[y_test == 1], ax=ax, color="#D9381E", fill=True, alpha=0.35, label="Latent Defect Components (1)", linewidth=2)
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=2, label=f"Calibrated Threshold = {threshold:.3f}")

    ax.set_title("Module A: Normalized Anomaly Score Distribution (Test Set)", pad=12, fontweight="bold", fontsize=13)
    ax.set_xlabel("Normalized Anomaly Score (0 = Nominal, 1 = Highly Anomalous)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(title="Component Ground Truth", loc="upper right")
    plt.tight_layout()
    fig.savefig(output_dir / "module_a_score_distribution.png", bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_anomaly_module(
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    output_dir: Path = Path("outputs"),
) -> Dict[str, Any]:
    """
    Execute end-to-end Module A training, calibration, and benchmarking.
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

    print(f"Dataset shapes: Train={df_train.shape}, Val={df_val.shape}, Test={df_test.shape}")

    # 1. Feature Engineering (strictly early features <= 24h)
    print("Fitting FeatureEngineer pipeline on training split...")
    fe = FeatureEngineer(stage=ESSStage.STAGE_24H, include_batch_relative=True)
    fe.fit(df_train, lot_column="Lot_ID")

    df_train_feat = fe.transform(df_train, lot_column="Lot_ID")
    df_val_feat = fe.transform(df_val, lot_column="Lot_ID")
    df_test_feat = fe.transform(df_test, lot_column="Lot_ID")

    y_train = df_train["Latent_Defect"].values
    y_val = df_val["Latent_Defect"].values
    y_test = df_test["Latent_Defect"].values

    # 2. Benchmark Statistical Baselines on Early Drift Rate
    print("\nEvaluating Statistical Baseline Detectors on Test Set...")
    z_detector = ZScoreDetector(threshold=3.0).fit(df_train_feat["Early_Drift_Rate_uA_per_h"])
    z_preds = z_detector.predict(df_test_feat["Early_Drift_Rate_uA_per_h"])
    z_recall = float(np.sum((y_test == 1) & (z_preds == 1)) / np.sum(y_test == 1))
    z_prec = float(np.sum((y_test == 1) & (z_preds == 1)) / max(1, np.sum(z_preds == 1)))

    rob_z_detector = RobustZScoreDetector(threshold=3.0).fit(df_train_feat["Early_Drift_Rate_uA_per_h"])
    rob_z_preds = rob_z_detector.predict(df_test_feat["Early_Drift_Rate_uA_per_h"])
    rob_z_recall = float(np.sum((y_test == 1) & (rob_z_preds == 1)) / np.sum(y_test == 1))
    rob_z_prec = float(np.sum((y_test == 1) & (rob_z_preds == 1)) / max(1, np.sum(rob_z_preds == 1)))

    iqr_detector = IQRDetector(k=1.5).fit(df_train_feat["Early_Drift_Rate_uA_per_h"])
    iqr_preds = iqr_detector.predict(df_test_feat["Early_Drift_Rate_uA_per_h"])
    iqr_recall = float(np.sum((y_test == 1) & (iqr_preds == 1)) / np.sum(y_test == 1))
    iqr_prec = float(np.sum((y_test == 1) & (iqr_preds == 1)) / max(1, np.sum(iqr_preds == 1)))

    print(f"  • Z-score Detector        : Recall = {z_recall:.2%}, Precision = {z_prec:.2%}")
    print(f"  • Robust Z-score Detector : Recall = {rob_z_recall:.2%}, Precision = {rob_z_prec:.2%}")
    print(f"  • IQR Fence Detector      : Recall = {iqr_recall:.2%}, Precision = {iqr_prec:.2%}")

    # 3. Train Dynamic Isolation Forest Anomaly Detector
    print("\nTraining DynamicAnomalyDetector (Isolation Forest)...")
    detector = DynamicAnomalyDetector(
        config={
            "n_estimators": 150,
            "contamination": 0.085,
            "max_samples": "auto",
            "random_state": 42,
            "n_jobs": -1,
        },
        feature_names=SELECTED_EARLY_FEATURES,
    )
    detector.fit(df_train_feat)

    # 4. Calibrate Decision Threshold on Validation Split
    print("Calibrating decision threshold on validation set (target Recall >= 98%)...")
    calibrated_threshold = detector.tune_threshold(df_val_feat, y_val, min_recall=0.98, preference="f1")
    print(f"Optimal calibrated threshold: {calibrated_threshold:.4f}")

    # 5. Evaluate All Splits
    train_metrics = detector.evaluate(df_train_feat, y_train)
    val_metrics = detector.evaluate(df_val_feat, y_val)
    test_metrics = detector.evaluate(df_test_feat, y_test)

    print("\n" + "=" * 60)
    print("MODULE A TEST SET EVALUATION RESULTS:")
    print("=" * 60)
    print(f"  • Test Components        : {len(df_test):,}")
    print(f"  • True Defective ICs     : {int(np.sum(y_test == 1)):,}")
    print(f"  • True Positives (TP)    : {test_metrics['confusion_matrix']['tp']:,}")
    print(f"  • True Negatives (TN)    : {test_metrics['confusion_matrix']['tn']:,}")
    print(f"  • False Positives (FP)   : {test_metrics['confusion_matrix']['fp']:,}")
    print(f"  • False Negatives (FN)   : {test_metrics['confusion_matrix']['fn']:,} (Escapes)")
    print(f"  • RECALL (Sensitivity)   : {test_metrics['recall']:.2%} (Baseline was 0.0%)")
    print(f"  • Precision              : {test_metrics['precision']:.2%}")
    print(f"  • F1 Score               : {test_metrics['f1_score']:.2%}")
    print(f"  • PR-AUC (Avg Precision) : {test_metrics['pr_auc']:.4f}")
    print(f"  • ROC-AUC                : {test_metrics['roc_auc']:.4f}")
    print(f"  • False Negative Rate    : {test_metrics['false_negative_rate_fnr']:.2%}")
    print("=" * 60)

    # 6. Save Model and Metadata
    model_path = models_dir / "anomaly_model.pkl"
    metadata_path = models_dir / "anomaly_metadata.json"

    print(f"\nSaving model to {model_path}...")
    detector.save(model_path)

    metadata = {
        "module": "Module A - Dynamic Anomaly Detection",
        "model_type": "IsolationForest",
        "library": "scikit-learn",
        "hyperparameters": detector.config,
        "selected_features": SELECTED_EARLY_FEATURES,
        "feature_count": len(SELECTED_EARLY_FEATURES),
        "calibrated_threshold": round(calibrated_threshold, 6),
        "statistical_baselines": {
            "z_score": {"recall": round(z_recall, 4), "precision": round(z_prec, 4)},
            "robust_z_score": {"recall": round(rob_z_recall, 4), "precision": round(rob_z_prec, 4)},
            "iqr_fence": {"recall": round(iqr_recall, 4), "precision": round(iqr_prec, 4)},
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        },
        "traditional_baseline_comparison": {
            "traditional_recall": 0.0,
            "module_a_recall": test_metrics["recall"],
            "traditional_escapes": int(np.sum(y_test == 1)),
            "module_a_escapes": test_metrics["silent_escapes"],
            "escapes_prevented_pct": round(
                ((int(np.sum(y_test == 1)) - test_metrics["silent_escapes"]) / int(np.sum(y_test == 1))) * 100.0, 2
            ),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}...")

    # 7. Generate Visualizations
    print("Generating evaluation visualization artifacts...")
    test_scores = detector.predict_anomaly_scores(df_test_feat)
    test_preds = detector.predict(df_test_feat)
    plot_module_a_visualizations(
        y_test=y_test,
        scores_test=test_scores,
        y_pred_test=test_preds,
        threshold=calibrated_threshold,
        metrics=test_metrics,
        output_dir=output_dir,
    )

    return metadata


def main():
    train_and_evaluate_anomaly_module()


if __name__ == "__main__":
    main()
