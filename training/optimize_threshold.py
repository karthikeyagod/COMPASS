"""
Phase 6: Anomaly Threshold & False-Negative Optimization for SIH Problem Statement 260170.

Performs fine-grained threshold sensitivity sweeps across validation and test sets,
optimizes operating thresholds for safety-critical screening (minimizing catastrophic
false negatives while maintaining practical false alarm rates), generates multi-panel
visualizations, evaluates latent defect escape recovery, and saves optimization artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.anomaly_detection import DynamicAnomalyDetector
from src.feature_engineering import ESSStage, FeatureEngineer


def compute_threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    fn_cost_weight: float = 100.0,
    fp_cost_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Compute comprehensive classification metrics across a spectrum of decision thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (1 = Latent Defect, 0 = Healthy).
    scores : np.ndarray
        Continuous normalized anomaly scores in [0, 1].
    thresholds : np.ndarray
        Array of threshold values to evaluate.
    fn_cost_weight : float
        Cost penalty assigned to a False Negative (catastrophic escape). Default 100.0.
    fp_cost_weight : float
        Cost penalty assigned to a False Positive (false alarm re-test). Default 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame containing sweep metrics for every threshold.
    """
    records = []
    total_pos = int(np.sum(y_true == 1))
    total_neg = int(np.sum(y_true == 0))

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        recall = tp / total_pos if total_pos > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fnr = fn / total_pos if total_pos > 0 else 0.0
        fpr = fp / total_neg if total_neg > 0 else 0.0
        cost = (fn * fn_cost_weight) + (fp * fp_cost_weight)

        records.append({
            "Threshold": round(float(t), 4),
            "True_Positives": tp,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "Recall": round(recall, 6),
            "Precision": round(precision, 6),
            "F1_Score": round(f1, 6),
            "False_Negative_Rate": round(fnr, 6),
            "False_Positive_Rate": round(fpr, 6),
            "Asymmetric_Cost": round(cost, 2),
        })

    return pd.DataFrame(records)


def select_optimal_threshold(
    sweep_df: pd.DataFrame,
    min_recall_target: float = 0.99,
    max_fpr_target: float = 0.02,
) -> Tuple[float, Dict[str, Any], str]:
    """
    Select optimal anomaly threshold using a safety-critical multi-objective rule:
    1. Primary constraint: Recall >= min_recall_target (minimizes catastrophic escapes).
    2. Secondary constraint: False Positive Rate <= max_fpr_target.
    3. Optimization objective: Maximize F1 Score / Minimize Asymmetric Cost.

    Returns
    -------
    Tuple[float, Dict[str, Any], str]
        (Selected threshold, row metrics dict, selection rationale text).
    """
    # Filter candidates meeting primary recall constraint
    candidates = sweep_df[sweep_df["Recall"] >= min_recall_target]

    if candidates.empty:
        # Fallback to maximum recall threshold
        best_idx = sweep_df["Recall"].idxmax()
        rationale = "Fallback: Selected threshold with absolute maximum recall available."
    else:
        # Filter candidates meeting FPR constraint
        valid_fpr = candidates[candidates["False_Positive_Rate"] <= max_fpr_target]
        if not valid_fpr.empty:
            best_idx = valid_fpr["F1_Score"].idxmax()
            rationale = (
                f"Selected optimal safety threshold meeting Recall >= {min_recall_target:.1%} "
                f"and FPR <= {max_fpr_target:.1%}, maximizing F1 Score."
            )
        else:
            # Pick minimum asymmetric cost among high recall candidates
            best_idx = candidates["Asymmetric_Cost"].idxmin()
            rationale = (
                f"Selected threshold meeting Recall >= {min_recall_target:.1%} "
                f"minimizing total asymmetric safety-critical mission cost."
            )

    selected_row = sweep_df.loc[best_idx].to_dict()
    selected_threshold = float(selected_row["Threshold"])
    return selected_threshold, selected_row, rationale


def plot_threshold_analysis(
    sweep_df: pd.DataFrame,
    selected_threshold: float,
    output_path: Path,
) -> None:
    """
    Render 4-panel publication-grade threshold sensitivity visualization.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    # 1. Precision vs. Recall Curve
    ax1 = axes[0, 0]
    ax1.plot(sweep_df["Recall"], sweep_df["Precision"], color="#2B5C8F", linewidth=2.5, label="PR Frontier")
    sel_row = sweep_df[np.isclose(sweep_df["Threshold"], selected_threshold, atol=0.005)].iloc[0]
    ax1.scatter(
        [sel_row["Recall"]],
        [sel_row["Precision"]],
        color="#D9381E",
        s=140,
        zorder=5,
        label=f"Selected Threshold ({selected_threshold:.3f})\nRecall={sel_row['Recall']:.1%}, Prec={sel_row['Precision']:.1%}",
    )
    ax1.set_title("1. Precision vs. Recall Trade-Off", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Recall (Detection Rate)", fontsize=10)
    ax1.set_ylabel("Precision (Defect Purity)", fontsize=10)
    ax1.set_xlim([0.0, 1.05])
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="lower left", fontsize=9.5)

    # 2. Threshold vs. Recall & False Negatives
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    l1 = ax2.plot(sweep_df["Threshold"], sweep_df["Recall"] * 100.0, color="#27AE60", linewidth=2.5, label="Recall (%)")
    l2 = ax2_twin.plot(sweep_df["Threshold"], sweep_df["False_Negatives"], color="#D9381E", linewidth=2.2, linestyle="--", label="False Negatives (Escapes)")
    ax2.axvline(selected_threshold, color="#333333", linestyle=":", linewidth=2, label=f"Selected Threshold = {selected_threshold:.3f}")
    ax2.set_title("2. Threshold vs. Recall & Catastrophic False Negatives", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Anomaly Decision Threshold", fontsize=10)
    ax2.set_ylabel("Recall (%)", color="#27AE60", fontsize=10)
    ax2_twin.set_ylabel("Count of False Negatives (Escapes)", color="#D9381E", fontsize=10)
    ax2_twin.grid(False)
    lines = l1 + l2 + [ax2.get_lines()[-1]]
    ax2.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=9.5)

    # 3. Threshold vs. Precision & False Positives
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    l3 = ax3.plot(sweep_df["Threshold"], sweep_df["Precision"] * 100.0, color="#2B5C8F", linewidth=2.5, label="Precision (%)")
    l4 = ax3_twin.plot(sweep_df["Threshold"], sweep_df["False_Positives"], color="#F39C12", linewidth=2.2, linestyle="--", label="False Positives (Re-test Overhead)")
    ax3.axvline(selected_threshold, color="#333333", linestyle=":", linewidth=2, label=f"Selected Threshold = {selected_threshold:.3f}")
    ax3.set_title("3. Threshold vs. Precision & Operational Re-Test Overhead", fontweight="bold", fontsize=12)
    ax3.set_xlabel("Anomaly Decision Threshold", fontsize=10)
    ax3.set_ylabel("Precision (%)", color="#2B5C8F", fontsize=10)
    ax3_twin.set_ylabel("Count of False Positives (False Alarms)", color="#F39C12", fontsize=10)
    ax3_twin.grid(False)
    lines_p = l3 + l4 + [ax3.get_lines()[-1]]
    ax3.legend(lines_p, [l.get_label() for l in lines_p], loc="center right", fontsize=9.5)

    # 4. Asymmetric Safety-Critical Mission Cost
    ax4 = axes[1, 1]
    ax4.plot(sweep_df["Threshold"], sweep_df["Asymmetric_Cost"], color="#8E44AD", linewidth=2.5, label="Total Mission Risk Cost (100x FN + 1x FP)")
    ax4.axvline(selected_threshold, color="#333333", linestyle=":", linewidth=2, label=f"Selected Threshold = {selected_threshold:.3f}")
    ax4.scatter([selected_threshold], [sel_row["Asymmetric_Cost"]], color="#D9381E", s=120, zorder=5, label=f"Optimal Risk Point\nCost = {sel_row['Asymmetric_Cost']:,.0f}")
    ax4.set_title("4. Asymmetric Safety-Critical Cost Function", fontweight="bold", fontsize=12)
    ax4.set_xlabel("Anomaly Decision Threshold", fontsize=10)
    ax4.set_ylabel("Total Penalty Cost (Arbitrary Units)", fontsize=10)
    ax4.legend(loc="upper left", fontsize=9.5)

    plt.suptitle("Module A: Decision Threshold Optimization & Sensitivity Analysis", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_threshold_optimization(
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    output_dir: Path = Path("outputs"),
) -> Dict[str, Any]:
    """
    Execute full Phase 6 optimization workflow.
    """
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)

    model_path = models_dir / "anomaly_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path.resolve()}. Run Phase 5 training first.")

    print(f"Loading trained Module A detector from {model_path}...")
    detector = DynamicAnomalyDetector.load(model_path)

    # Load datasets
    df_train = pd.read_csv(data_dir / "sih_260170_train.csv")
    df_val = pd.read_csv(data_dir / "sih_260170_validation.csv")
    df_test = pd.read_csv(data_dir / "sih_260170_test.csv")
    df_full = pd.read_csv(data_dir / "sih_260170_synthetic_ess_dataset.csv")

    # Feature Engineering
    print("Extracting early features across all partitions...")
    fe = FeatureEngineer(stage=ESSStage.STAGE_24H, include_batch_relative=True)
    fe.fit(df_train, lot_column="Lot_ID")

    df_val_feat = fe.transform(df_val, lot_column="Lot_ID")
    df_test_feat = fe.transform(df_test, lot_column="Lot_ID")
    df_full_feat = fe.transform(df_full, lot_column="Lot_ID")

    y_val = df_val["Latent_Defect"].values
    y_test = df_test["Latent_Defect"].values
    y_full = df_full["Latent_Defect"].values

    val_scores = detector.predict_anomaly_scores(df_val_feat)
    test_scores = detector.predict_anomaly_scores(df_test_feat)
    full_scores = detector.predict_anomaly_scores(df_full_feat)

    # 1. 200-Point Fine-Grained Threshold Sweep on Test Set
    print("\nExecuting 200-point threshold sensitivity sweep on test dataset...")
    thresholds = np.linspace(0.01, 0.99, 200)
    sweep_df = compute_threshold_sweep(y_test, test_scores, thresholds, fn_cost_weight=100.0, fp_cost_weight=1.0)

    csv_output_path = output_dir / "anomaly_threshold_analysis.csv"
    sweep_df.to_csv(csv_output_path, index=False)
    print(f"Saved threshold sweep analysis table to: {csv_output_path.resolve()}")

    # 2. Optimal Threshold Selection Rule (calibrated on validation split)
    val_sweep_df = compute_threshold_sweep(y_val, val_scores, thresholds)
    selected_threshold, val_metrics, rationale = select_optimal_threshold(
        val_sweep_df,
        min_recall_target=0.99,
        max_fpr_target=0.02,
    )
    print(f"\nSelected Optimal Operating Threshold: {selected_threshold:.4f}")
    print(f"Selection Rationale: {rationale}")

    # Update detector threshold
    detector.threshold = selected_threshold
    detector.save(model_path)

    # 3. Multi-Panel Visualization
    plot_output_path = output_dir / "anomaly_threshold_plot.png"
    print(f"Generating threshold analysis plots to {plot_output_path}...")
    plot_threshold_analysis(sweep_df, selected_threshold, plot_output_path)

    # 4. Comprehensive Evaluation at Selected Threshold
    test_eval = detector.evaluate(df_test_feat, y_test, threshold=selected_threshold)
    full_eval = detector.evaluate(df_full_feat, y_full, threshold=selected_threshold)

    # 5. SPECIAL EVALUATION: Latent defects passing datasheet limit but caught by Module A
    # Ground truth: Latent_Defect == 1 AND Value_168h <= Datasheet_Limit AND Module_A == 1
    datasheet_limit = df_full["Datasheet_Leakage_Limit_uA"]
    full_pred = detector.predict(df_full_feat, threshold=selected_threshold)
    
    # Components that pass static limit
    static_pass_mask = df_full["Value_168h_uA"] <= datasheet_limit
    
    # Latent defect components that pass static limit
    latent_defect_static_pass = (df_full["Latent_Defect"] == 1) & static_pass_mask
    total_silent_escapes_baseline = int(np.sum(latent_defect_static_pass))
    
    # Latent defects that pass static limit BUT are caught by Module A
    caught_by_module_a = latent_defect_static_pass & (full_pred == 1)
    escapes_caught_count = int(np.sum(caught_by_module_a))
    escapes_recovery_rate_pct = (escapes_caught_count / total_silent_escapes_baseline) * 100.0

    # Test split specific escape recovery
    test_pred = detector.predict(df_test_feat, threshold=selected_threshold)
    test_static_pass = df_test["Value_168h_uA"] <= df_test["Datasheet_Leakage_Limit_uA"]
    test_latent_static_pass = (df_test["Latent_Defect"] == 1) & test_static_pass
    test_escapes_baseline = int(np.sum(test_latent_static_pass))
    test_escapes_caught = int(np.sum(test_latent_static_pass & (test_pred == 1)))
    test_recovery_pct = (test_escapes_caught / test_escapes_baseline) * 100.0

    print("\n" + "=" * 65)
    print("SPECIAL EVALUATION: RECOVERING SILENT ESCAPES VIA MODULE A")
    print("=" * 65)
    print(f"  • Total Tested Components (Full Dataset)   : {len(df_full):,}")
    print(f"  • Total Latent Defect Components           : {int(np.sum(df_full['Latent_Defect'] == 1)):,}")
    print(f"  • Defective ICs Passing Static Limit       : {total_silent_escapes_baseline:,} (100.00% Baseline Escape)")
    print(f"  • DEFECTS DETECTED BY MODULE A AT 24H      : {escapes_caught_count:,} / {total_silent_escapes_baseline:,}")
    print(f"  • ESCAPES RECOVERY RATE                    : {escapes_recovery_rate_pct:.2f}%")
    print("-" * 65)
    print(f"  • Test Split Silent Escapes Baseline       : {test_escapes_baseline:,} / {test_escapes_baseline:,} (100.00% missed)")
    print(f"  • Test Split Defects Detected by Module A  : {test_escapes_caught:,} / {test_escapes_baseline:,}")
    print(f"  • Test Split Recovery Rate                 : {test_recovery_pct:.2f}% (Only {test_escapes_baseline - test_escapes_caught} escape remains)")
    print("=" * 65)

    # 6. Save models/anomaly_threshold.json
    threshold_metadata = {
        "module": "Module A - Threshold & False Negative Optimization",
        "selected_threshold": selected_threshold,
        "selection_rule": {
            "min_recall_constraint": 0.99,
            "max_fpr_constraint": 0.02,
            "objective": "Maximize F1 Score subject to safety-critical escape elimination",
            "rationale": rationale,
        },
        "test_set_performance": test_eval,
        "full_dataset_performance": full_eval,
        "special_sih_evaluation": {
            "description": "Latent defects that pass traditional datasheet limit (<= 50 uA) but are detected by Module A at 24h",
            "full_dataset": {
                "total_latent_defects": int(np.sum(df_full["Latent_Defect"] == 1)),
                "defects_passing_datasheet_limit": total_silent_escapes_baseline,
                "defects_detected_by_module_a": escapes_caught_count,
                "defects_missed_by_module_a": total_silent_escapes_baseline - escapes_caught_count,
                "escape_recovery_percentage": round(escapes_recovery_rate_pct, 2),
            },
            "test_split": {
                "total_latent_defects": test_escapes_baseline,
                "defects_passing_datasheet_limit": test_escapes_baseline,
                "defects_detected_by_module_a": test_escapes_caught,
                "defects_missed_by_module_a": test_escapes_baseline - test_escapes_caught,
                "escape_recovery_percentage": round(test_recovery_pct, 2),
            },
        },
    }

    threshold_json_path = models_dir / "anomaly_threshold.json"
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_metadata, f, indent=2)
    print(f"Saved threshold metadata to: {threshold_json_path.resolve()}")

    return threshold_metadata


def main():
    run_threshold_optimization()


if __name__ == "__main__":
    main()
