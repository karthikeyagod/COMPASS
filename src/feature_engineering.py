"""
Feature Engineering Module for SIH Problem Statement 260170.

Transforms raw Environmental Stress Screening (ESS) physical measurements into:
1. Dynamic early drift kinetic indicators (0h -> 24h).
2. Batch/Peer-relative lot statistics (Mean, Median, STD, IQR, Robust Z-Score, Percentile).
3. Multi-channel degradation indicators (Leakage current, Propagation delay, Iddq).
4. Retrospective full-cycle benchmark features (96h, 168h).

Strictly enforces data leakage prevention between early screening (<=24h) and
retrospective full-cycle burn-in (>24h).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd


class ESSStage(str, Enum):
    """Enumeration of screening timeline stages during ESS."""
    STAGE_0H = "0h"      # Pre-screening baseline
    STAGE_24H = "24h"    # Early screening checkpoint (Decision point for Module A & B)
    STAGE_96H = "96h"    # Mid-cycle stress checkpoint
    STAGE_168H = "168h"  # Full qualification completion (Retrospective truth)


class DataLeakageError(ValueError):
    """Raised when features from later ESS stages are used in earlier stages."""
    pass


# Feature availability mapping by ESS stage
STAGE_FEATURE_REGISTRY: Dict[ESSStage, List[str]] = {
    ESSStage.STAGE_0H: [
        "Value_0h_uA",
        "Iddq_0h_mA",
        "PropagationDelay_0h_ns",
        "Temperature_C",
    ],
    ESSStage.STAGE_24H: [
        "Value_0h_uA",
        "Value_24h_uA",
        "Early_Drift_uA",
        "Early_Drift_Rate_uA_per_h",
        "Early_Percentage_Change",
        "Iddq_0h_mA",
        "Iddq_24h_mA",
        "Early_Iddq_Drift_mA",
        "Early_Iddq_Percentage_Change",
        "PropagationDelay_0h_ns",
        "PropagationDelay_24h_ns",
        "Early_Delay_Drift_ns",
        "Early_Delay_Drift_Rate_ns_per_h",
        "Early_Delay_Percentage_Change",
        "Temperature_C",
    ],
    ESSStage.STAGE_96H: [
        "Value_96h_uA",
        "Drift_96h_uA",
        "Drift_Rate_96h_uA_per_h",
        "PropagationDelay_96h_ns",
        "Delay_Drift_96h_ns",
    ],
    ESSStage.STAGE_168H: [
        "Value_168h_uA",
        "Total_Drift_uA",
        "Total_Drift_Rate_uA_per_h",
        "Total_Percentage_Change",
        "Late_Acceleration_Ratio",
        "PropagationDelay_168h_ns",
        "Delay_Total_Drift_ns",
        "Delay_Total_Percentage_Change",
        "Datasheet_Leakage_Limit_uA",
        "Absolute_Limit_Pass",
    ],
}

# Features that MUST NOT be used for early screening (t <= 24h)
RETROSPECTIVE_RAW_COLS = [
    "Value_96h_uA",
    "Value_168h_uA",
    "PropagationDelay_96h_ns",
    "PropagationDelay_168h_ns",
    "Absolute_Limit_Pass",
]

RETROSPECTIVE_DERIVED_COLS = [
    "Drift_96h_uA",
    "Drift_Rate_96h_uA_per_h",
    "Delay_Drift_96h_ns",
    "Total_Drift_uA",
    "Total_Drift_Rate_uA_per_h",
    "Total_Percentage_Change",
    "Late_Acceleration_Ratio",
    "Delay_Total_Drift_ns",
    "Delay_Total_Percentage_Change",
]


def validate_feature_availability(
    features: Sequence[str],
    current_stage: Union[str, ESSStage] = ESSStage.STAGE_24H,
) -> None:
    """
    Validate that none of the requested features violate time-series causality
    or cause data leakage for the specified ESS stage.

    Parameters
    ----------
    features : Sequence[str]
        List of feature column names to validate.
    current_stage : Union[str, ESSStage]
        The current screening stage ('0h', '24h', '96h', '168h').

    Raises
    ------
    DataLeakageError
        If a retrospective feature is present in an earlier stage feature set.
    """
    stage_enum = ESSStage(current_stage) if isinstance(current_stage, str) else current_stage

    if stage_enum in (ESSStage.STAGE_0H, ESSStage.STAGE_24H):
        leaked = [f for f in features if f in RETROSPECTIVE_RAW_COLS or f in RETROSPECTIVE_DERIVED_COLS]
        if leaked:
            raise DataLeakageError(
                f"Data Leakage Violation: The following retrospective features are unavailable "
                f"at stage '{stage_enum.value}' and cannot be used for early prediction: {leaked}"
            )


def compute_early_drift_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute early kinetic drift and multi-channel rate-of-change features (t <= 24h).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 0h and 24h telemetry columns.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame with early drift features appended.
    """
    df_out = df.copy()
    eps = 1e-9

    # 1. Leakage Current Early Kinetics (Value_0h -> Value_24h)
    if "Value_0h_uA" in df_out.columns and "Value_24h_uA" in df_out.columns:
        df_out["Early_Drift_uA"] = df_out["Value_24h_uA"] - df_out["Value_0h_uA"]
        df_out["Early_Drift_Rate_uA_per_h"] = df_out["Early_Drift_uA"] / 24.0
        df_out["Early_Percentage_Change"] = (
            (df_out["Early_Drift_uA"] / (df_out["Value_0h_uA"].abs() + eps)) * 100.0
        )

    # 2. Propagation Delay Early Kinetics (0h -> 24h)
    if "PropagationDelay_0h_ns" in df_out.columns and "PropagationDelay_24h_ns" in df_out.columns:
        df_out["Early_Delay_Drift_ns"] = (
            df_out["PropagationDelay_24h_ns"] - df_out["PropagationDelay_0h_ns"]
        )
        df_out["Early_Delay_Drift_Rate_ns_per_h"] = df_out["Early_Delay_Drift_ns"] / 24.0
        df_out["Early_Delay_Percentage_Change"] = (
            (df_out["Early_Delay_Drift_ns"] / (df_out["PropagationDelay_0h_ns"].abs() + eps)) * 100.0
        )

    # 3. Quiescent Current Early Kinetics (Iddq 0h -> 24h)
    if "Iddq_0h_mA" in df_out.columns and "Iddq_24h_mA" in df_out.columns:
        df_out["Early_Iddq_Drift_mA"] = df_out["Iddq_24h_mA"] - df_out["Iddq_0h_mA"]
        df_out["Early_Iddq_Percentage_Change"] = (
            (df_out["Early_Iddq_Drift_mA"] / (df_out["Iddq_0h_mA"].abs() + eps)) * 100.0
        )

    return df_out


def compute_batch_relative_features(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    lot_column: str = "Lot_ID",
    stats_dict: Optional[Dict[str, pd.DataFrame]] = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compute manufacturing batch/lot-relative peer comparison metrics:
    - Lot_Mean
    - Lot_Median
    - Lot_STD
    - Lot_IQR
    - Robust_Z_Score: (x - Lot_Median) / (Lot_IQR / 1.34898)
    - Lot_Z_Score: (x - Lot_Mean) / Lot_STD
    - Percentile: Intra-lot percentile rank (0 to 100%)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_columns : Optional[List[str]]
        Columns on which to calculate lot statistics. Default uses key early features.
    lot_column : str
        Column denoting manufacturing lot. Default 'Lot_ID'.
    stats_dict : Optional[Dict[str, pd.DataFrame]]
        Precomputed lot statistics (for inference / test sets to prevent leakage).

    Returns
    -------
    tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
        (Transformed DataFrame, dictionary of computed lot statistics per feature).
    """
    if lot_column not in df.columns:
        return df.copy(), {}

    if target_columns is None:
        target_columns = [
            col for col in [
                "Value_0h_uA",
                "Value_24h_uA",
                "Early_Drift_uA",
                "Early_Drift_Rate_uA_per_h",
                "Early_Percentage_Change",
                "PropagationDelay_0h_ns",
                "Early_Delay_Drift_ns",
            ] if col in df.columns
        ]

    df_out = df.copy()
    computed_stats: Dict[str, pd.DataFrame] = {}
    eps = 1e-7

    for col in target_columns:
        # Compute lot statistics from current df
        grouped = df_out.groupby(lot_column)[col]
        curr_mean = grouped.mean()
        curr_median = grouped.median()
        curr_std = grouped.std(ddof=1).fillna(0.0)
        curr_q25 = grouped.quantile(0.25)
        curr_q75 = grouped.quantile(0.75)
        curr_iqr = (curr_q75 - curr_q25).clip(lower=0.0)

        global_mean = float(df_out[col].mean())
        global_median = float(df_out[col].median())
        global_std = float(df_out[col].std(ddof=1)) if len(df_out) > 1 else 0.0
        global_iqr = float(df_out[col].quantile(0.75) - df_out[col].quantile(0.25))

        if stats_dict and col in stats_dict:
            base_stat = stats_dict[col]
            mean_series = base_stat["mean"]
            median_series = base_stat["median"]
            std_series = base_stat["std"]
            iqr_series = base_stat["iqr"]
        else:
            base_stat = pd.DataFrame({
                "mean": curr_mean,
                "median": curr_median,
                "std": curr_std,
                "iqr": curr_iqr,
            })
            computed_stats[col] = base_stat
            mean_series = curr_mean
            median_series = curr_median
            std_series = curr_std
            iqr_series = curr_iqr

        # Map lot statistics to rows with fallback for unseen lots
        lot_series = df_out[lot_column]
        mean_mapped = lot_series.map(mean_series).fillna(lot_series.map(curr_mean)).fillna(global_mean).astype(float)
        median_mapped = lot_series.map(median_series).fillna(lot_series.map(curr_median)).fillna(global_median).astype(float)
        std_mapped = lot_series.map(std_series).fillna(lot_series.map(curr_std)).fillna(global_std).astype(float)
        iqr_mapped = lot_series.map(iqr_series).fillna(lot_series.map(curr_iqr)).fillna(global_iqr).astype(float)

        col_clean = col.replace("_uA", "").replace("_ns", "").replace("_mA", "")

        df_out[f"{col_clean}_Lot_Mean"] = mean_mapped
        df_out[f"{col_clean}_Lot_Median"] = median_mapped
        df_out[f"{col_clean}_Lot_STD"] = std_mapped
        df_out[f"{col_clean}_Lot_IQR"] = iqr_mapped

        # Normal scale factor for IQR = 1.34898
        norm_iqr = (iqr_mapped / 1.34898).replace(0.0, np.nan).fillna(std_mapped).replace(0.0, eps)
        df_out[f"{col_clean}_Robust_Z_Score"] = (df_out[col] - median_mapped) / (norm_iqr + eps)
        df_out[f"{col_clean}_Lot_Z_Score"] = (df_out[col] - mean_mapped) / (std_mapped.replace(0.0, eps) + eps)

        # Intra-lot percentile rank (0 to 100)
        df_out[f"{col_clean}_Lot_Percentile"] = df_out.groupby(lot_column)[col].rank(pct=True) * 100.0

    return df_out, (stats_dict or computed_stats)


def compute_retrospective_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute retrospective post-burn-in features (t > 24h: 96h, 168h).
    WARNING: For post-screening validation and analysis only.
    DO NOT pass these into early anomaly detection (Module A) or drift forecasting (Module B).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 96h and 168h columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with retrospective features appended.
    """
    df_out = df.copy()
    eps = 1e-9

    if "Value_96h_uA" in df_out.columns and "Value_0h_uA" in df_out.columns:
        df_out["Drift_96h_uA"] = df_out["Value_96h_uA"] - df_out["Value_0h_uA"]
        df_out["Drift_Rate_96h_uA_per_h"] = df_out["Drift_96h_uA"] / 96.0

    if "Value_168h_uA" in df_out.columns and "Value_0h_uA" in df_out.columns:
        df_out["Total_Drift_uA"] = df_out["Value_168h_uA"] - df_out["Value_0h_uA"]
        df_out["Total_Drift_Rate_uA_per_h"] = df_out["Total_Drift_uA"] / 168.0
        df_out["Total_Percentage_Change"] = (
            (df_out["Total_Drift_uA"] / (df_out["Value_0h_uA"].abs() + eps)) * 100.0
        )

    if "Value_168h_uA" in df_out.columns and "Value_96h_uA" in df_out.columns and "Value_24h_uA" in df_out.columns:
        late_drift = df_out["Value_168h_uA"] - df_out["Value_96h_uA"]
        early_drift = (df_out["Value_24h_uA"] - df_out["Value_0h_uA"]).abs() + eps
        df_out["Late_Acceleration_Ratio"] = late_drift / early_drift

    if "PropagationDelay_168h_ns" in df_out.columns and "PropagationDelay_0h_ns" in df_out.columns:
        df_out["Delay_Total_Drift_ns"] = (
            df_out["PropagationDelay_168h_ns"] - df_out["PropagationDelay_0h_ns"]
        )
        df_out["Delay_Total_Percentage_Change"] = (
            (df_out["Delay_Total_Drift_ns"] / (df_out["PropagationDelay_0h_ns"].abs() + eps)) * 100.0
        )

    return df_out


class FeatureEngineer:
    """
    Stateful feature engineering pipeline supporting fitting on training lots,
    transforming unseen validation/test streams, and stage-aware feature extraction.
    """

    def __init__(
        self,
        stage: Union[str, ESSStage] = ESSStage.STAGE_24H,
        include_batch_relative: bool = True,
        target_lot_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the FeatureEngineer.

        Parameters
        ----------
        stage : Union[str, ESSStage]
            Maximum permitted ESS timeline stage (default '24h').
        include_batch_relative : bool
            Whether to compute lot-relative peer statistics.
        target_lot_cols : Optional[List[str]]
            Specific columns for lot peer comparison.
        """
        self.stage = ESSStage(stage) if isinstance(stage, str) else stage
        self.include_batch_relative = include_batch_relative
        self.target_lot_cols = target_lot_cols
        self.lot_stats_: Dict[str, pd.DataFrame] = {}
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, lot_column: str = "Lot_ID") -> "FeatureEngineer":
        """
        Fit lot-level peer statistical distributions on a training set.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.
        lot_column : str
            Lot ID column name.

        Returns
        -------
        FeatureEngineer
            Fitted instance.
        """
        df_early = compute_early_drift_features(df)
        if self.include_batch_relative and lot_column in df.columns:
            _, self.lot_stats_ = compute_batch_relative_features(
                df_early,
                target_columns=self.target_lot_cols,
                lot_column=lot_column,
            )
        return self

    def transform(self, df: pd.DataFrame, lot_column: str = "Lot_ID") -> pd.DataFrame:
        """
        Transform raw measurements into stage-compliant engineered feature set.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        lot_column : str
            Lot ID column name.

        Returns
        -------
        pd.DataFrame
            Engineered feature matrix.
        """
        # Step 1: Compute early drift features
        df_out = compute_early_drift_features(df)

        # Step 2: Compute lot peer features
        if self.include_batch_relative and lot_column in df_out.columns:
            df_out, _ = compute_batch_relative_features(
                df_out,
                target_columns=self.target_lot_cols,
                lot_column=lot_column,
                stats_dict=self.lot_stats_,
            )

        # Step 3: Compute retrospective features ONLY if stage allows (> 24h)
        if self.stage in (ESSStage.STAGE_96H, ESSStage.STAGE_168H):
            df_out = compute_retrospective_features(df_out)

        return df_out

    def fit_transform(self, df: pd.DataFrame, lot_column: str = "Lot_ID") -> pd.DataFrame:
        """
        Fit and transform on the training set.
        """
        return self.fit(df, lot_column=lot_column).transform(df, lot_column=lot_column)

    def get_early_feature_names(self, df_transformed: pd.DataFrame) -> List[str]:
        """
        Extract strictly early-stage feature column names suitable for Module A and B.

        Parameters
        ----------
        df_transformed : pd.DataFrame
            Transformed DataFrame.

        Returns
        -------
        List[str]
            List of valid early feature column names.
        """
        exclude = {
            "Component_ID",
            "Lot_ID",
            "Split",
            "Latent_Defect",
            "Datasheet_Leakage_Limit_uA",
            "Absolute_Limit_Pass",
        } | set(RETROSPECTIVE_RAW_COLS) | set(RETROSPECTIVE_DERIVED_COLS)

        early_cols = [c for c in df_transformed.columns if c not in exclude]
        validate_feature_availability(early_cols, current_stage=ESSStage.STAGE_24H)
        return early_cols
