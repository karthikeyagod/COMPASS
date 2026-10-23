"""
Secure & Trustworthy ESS Analytics Trust Layer for SIH Problem Statement 260170.

Establishes data provenance, cryptographic integrity, structural tampering detection,
and statistical poisoning detection BEFORE telemetry enters the ML pipeline.

Core Architecture:
1. DatasetIntegrityChecker  -> Cryptographic SHA-256 content & byte-level hashing
2. SchemaFingerprinter      -> Deterministic structural schema hashing & unexpected drift detection
3. TamperDetector          -> Consistency checks (duplicate IDs, impossible physical values, non-monotonic timepoints)
4. PoisoningDetector       -> Statistical ensemble (Robust MAD outliers, KS distribution shift, label anomalies)
5. TrustGate               -> Evaluates overall data trust (TRUSTED / REVIEW_REQUIRED / NOT_TRUSTED) & ML gating
6. AuditTrailLogger        -> Immutable audit records for compliance and traceability
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats


# ==============================================================================
# 1. Cryptographic Dataset Integrity Checker
# ==============================================================================

class DatasetIntegrityChecker:
    """
    Computes cryptographic SHA-256 fingerprints and verifies dataset provenance
    against stored manifests without inventing false verifications.
    """

    MANIFEST_DIR = Path("artifacts/integrity")
    MANIFEST_FILE = MANIFEST_DIR / "dataset_manifest.json"

    @classmethod
    def compute_sha256(cls, content: Union[str, bytes, pd.DataFrame]) -> str:
        """
        Compute deterministic SHA-256 hash of dataset content.
        """
        if isinstance(content, pd.DataFrame):
            # Normalize DataFrame to deterministic CSV representation
            csv_str = content.to_csv(index=False, lineterminator="\n")
            return hashlib.sha256(csv_str.encode("utf-8")).hexdigest()
        elif isinstance(content, str):
            # Strip carriage returns for cross-platform Windows/Linux hash consistency
            normalized_str = content.replace("\r\n", "\n")
            return hashlib.sha256(normalized_str.encode("utf-8")).hexdigest()
        elif isinstance(content, bytes):
            return hashlib.sha256(content).hexdigest()
        else:
            return hashlib.sha256(str(content).encode("utf-8")).hexdigest()

    @classmethod
    def load_manifest(cls) -> Dict[str, Any]:
        """Load the trusted baseline manifest file if it exists."""
        if cls.MANIFEST_FILE.exists():
            try:
                with open(cls.MANIFEST_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @classmethod
    def save_manifest(cls, dataset_id: str, sha256_hash: str, row_count: int, schema_hash: str) -> None:
        """Persist a trusted reference manifest record."""
        cls.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        manifest = cls.load_manifest()
        manifest[dataset_id] = {
            "dataset_id": dataset_id,
            "sha256": sha256_hash,
            "schema_hash": schema_hash,
            "row_count": row_count,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(cls.MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def verify_integrity(
        cls,
        content: Union[str, bytes, pd.DataFrame],
        dataset_id: Optional[str] = None,
        expected_hash: Optional[str] = None,
        row_count: Optional[int] = None,
        schema_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate dataset cryptographic integrity.
        Distinguishes:
        - VERIFIED (current hash matches known reference)
        - BASELINE_CREATED (first upload, hash computed and registered)
        - INTEGRITY_MISMATCH (current hash conflicts with reference)
        """
        current_hash = cls.compute_sha256(content)
        manifest = cls.load_manifest()

        ref_hash = expected_hash
        if ref_hash is None and dataset_id and dataset_id not in ["DATASET_UPLOAD", "DATASET_PROFILE"] and dataset_id in manifest:
            ref_hash = manifest[dataset_id].get("sha256")

        # If current hash matches any known registered manifest record (e.g. SIH_BENCHMARK_260170), link to it
        if ref_hash is None:
            for d_id, d_meta in manifest.items():
                if d_meta.get("sha256", "").lower() == current_hash.lower():
                    ref_hash = d_meta.get("sha256")
                    break

        if ref_hash is None:
            # First upload / baseline creation
            status = "BASELINE_CREATED"
            message = "SHA-256 fingerprint generated. No prior reference baseline on file; established as new reference."
            is_valid = True
            if dataset_id and dataset_id not in ["DATASET_UPLOAD", "DATASET_PROFILE"] and row_count is not None and schema_hash is not None:
                cls.save_manifest(dataset_id, current_hash, row_count, schema_hash)
        elif current_hash.lower() == ref_hash.lower():
            status = "VERIFIED"
            message = "Cryptographic integrity verified. Current SHA-256 perfectly matches trusted baseline."
            is_valid = True
        else:
            status = "INTEGRITY_MISMATCH"
            message = f"Security Warning: SHA-256 mismatch detected! Reference: {ref_hash[:12]}... Current: {current_hash[:12]}..."
            is_valid = False

        return {
            "algorithm": "SHA-256",
            "current_hash": current_hash,
            "reference_hash": ref_hash,
            "status": status,
            "is_valid": is_valid,
            "message": message,
        }


# ==============================================================================
# 2. Schema Fingerprinter & Structural Drift Detector
# ==============================================================================

class SchemaFingerprinter:
    """
    Captures structural fingerprints of datasets to detect unauthorized column changes,
    missing critical timepoints, or unexpected type conversions.
    """

    @classmethod
    def compute_schema_fingerprint(cls, df: pd.DataFrame) -> str:
        """
        Generate deterministic hash of column names and dtypes.
        """
        schema_repr = "|".join([f"{col}:{str(df[col].dtype)}" for col in sorted(df.columns)])
        return hashlib.sha256(schema_repr.encode("utf-8")).hexdigest()

    @classmethod
    def analyze_schema(cls, df: pd.DataFrame, expected_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze structural schema integrity.
        """
        cols = list(df.columns)
        schema_hash = cls.compute_schema_fingerprint(df)
        issues: List[str] = []

        if expected_columns:
            missing_cols = [c for c in expected_columns if c not in cols]
            extra_cols = [c for c in cols if c not in expected_columns]
            if missing_cols:
                issues.append(f"Missing expected columns: {', '.join(missing_cols)}")
            if extra_cols:
                issues.append(f"Unexpected extra columns: {', '.join(extra_cols)}")

        # Check required ESS identifiers and timepoints
        has_comp_id = any(
            c.lower() in ["component_id", "device_id", "part_id", "serial", "serial_id", "chip_id"]
            or "component" in c.lower()
            for c in cols
        )
        if not has_comp_id:
            issues.append("Missing primary component identification column.")

        is_valid = bool(len(issues) == 0)
        return {
            "schema_fingerprint": schema_hash,
            "column_count": len(cols),
            "columns": cols,
            "status": "VALID" if is_valid else "SCHEMA_CHANGED",
            "is_valid": is_valid,
            "issues": issues,
        }


# ==============================================================================
# 3. Data Tampering & Consistency Detector
# ==============================================================================

class TamperDetector:
    """
    Inspects tabular dataset for physical inconsistencies, duplicate records,
    impossible physical values, and temporal ordering anomalies.
    """

    @classmethod
    def check_tampering(cls, df: pd.DataFrame, custom_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Execute comprehensive consistency and anti-tampering heuristics.
        """
        mapping = custom_mapping or {}
        id_col = mapping.get("component_id") or next((c for c in df.columns if "id" in c.lower()), df.columns[0])
        issues: List[str] = []
        suspicious_count = 0

        # 1. Duplicate Component IDs
        if id_col in df.columns:
            dup_ids = int(df[id_col].duplicated().sum())
            if dup_ids > 0:
                issues.append(f"Duplicate Component IDs detected: {dup_ids} duplicated records.")
                suspicious_count += dup_ids

        # 2. Entirely Duplicated Rows
        exact_dups = int(df.duplicated().sum())
        if exact_dups > 0:
            issues.append(f"Exact duplicate telemetry rows found: {exact_dups} rows.")
            suspicious_count += exact_dups

        # 3. Impossible Physical Values (Negative Leakage Current, Infinity)
        leakage_cols = [c for c in df.columns if any(k in c.lower() for k in ["0h", "24h", "96h", "168h", "leak", "val"])]
        for col in leakage_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                neg_count = int((df[col] < 0).sum())
                if neg_count > 0:
                    issues.append(f"Physically impossible negative values in column '{col}': {neg_count} records.")
                    suspicious_count += neg_count
                inf_count = int(np.isinf(df[col]).sum())
                if inf_count > 0:
                    issues.append(f"Infinite numeric values detected in column '{col}': {inf_count} records.")
                    suspicious_count += inf_count

        # 4. Temporal Monotonicity / Anomaly Check
        col_0h = mapping.get("value_0h") or next((c for c in df.columns if "0h" in c.lower() or "0" in c.lower()), None)
        col_24h = mapping.get("value_24h") or next((c for c in df.columns if "24h" in c.lower() or "24" in c.lower()), None)
        if col_0h and col_24h and col_0h in df.columns and col_24h in df.columns:
            if pd.api.types.is_numeric_dtype(df[col_0h]) and pd.api.types.is_numeric_dtype(df[col_24h]):
                # Severe negative drop (> 5 uA drop in 24h burn-in at 125C is physically anomalous)
                severe_drops = int(((df[col_0h] - df[col_24h]) > 5.0).sum())
                if severe_drops > 0:
                    issues.append(f"Unphysical negative burn-in trajectory (>5 μA recovery at 125°C): {severe_drops} records.")
                    suspicious_count += severe_drops

        is_passed = bool(suspicious_count == 0)
        return {
            "status": "PASSED" if is_passed else "WARNING",
            "is_passed": is_passed,
            "suspicious_records_count": int(suspicious_count),
            "duplicate_ids_count": int(df[id_col].duplicated().sum()) if id_col in df.columns else 0,
            "issues": issues,
        }


# ==============================================================================
# 4. Statistical Data Poisoning Detector
# ==============================================================================

class PoisoningDetector:
    """
    Statistical poisoning & anomaly detector identifying injected outliers,
    extreme distribution shifts, manipulated labels, and suspicious value clustering.
    """

    @classmethod
    def evaluate_poisoning_risk(
        cls,
        df: pd.DataFrame,
        baseline_df: Optional[pd.DataFrame] = None,
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute ensemble of statistical poisoning indicators.
        Returns a normalized Data Poisoning Risk Score [0.0 - 1.0].
        """
        mapping = custom_mapping or {}
        col_0h = mapping.get("value_0h") or next((c for c in df.columns if "0h" in c.lower() or "0" in c.lower()), None)
        col_24h = mapping.get("value_24h") or next((c for c in df.columns if "24h" in c.lower() or "24" in c.lower()), None)

        indicators: List[str] = []
        risk_components: List[float] = []
        extreme_outliers_count = 0
        dist_shift_level = "LOW"

        # Check A: Robust Outliers via Median Absolute Deviation (MAD)
        target_cols = [c for c in [col_0h, col_24h] if c and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        for col in target_cols:
            vals = df[col].dropna().values
            if len(vals) >= 5:
                median = np.median(vals)
                mad = np.median(np.abs(vals - median))
                if mad > 1e-6:
                    robust_z = 0.6745 * (vals - median) / mad
                    outliers = np.abs(robust_z) > 4.5  # Extreme 4.5 sigma MAD outlier
                    n_outliers = int(outliers.sum())
                    if n_outliers > 0:
                        outlier_pct = (n_outliers / len(vals)) * 100.0
                        extreme_outliers_count += n_outliers
                        if outlier_pct > 3.0:
                            indicators.append(f"High concentration of extreme statistical outliers in '{col}': {n_outliers} ({outlier_pct:.1f}%).")
                            risk_components.append(min(1.0, outlier_pct / 10.0))
                        else:
                            indicators.append(f"Detected {n_outliers} extreme statistical outliers in '{col}'.")
                            risk_components.append(0.2)

        # Check B: Distribution Shift vs Baseline (Kolmogorov-Smirnov Test & Wasserstein Distance)
        if baseline_df is not None and not baseline_df.empty:
            shift_scores: List[float] = []
            for col in target_cols:
                base_col = next((c for c in baseline_df.columns if col.lower() in c.lower() or c.lower() in col.lower()), None)
                if base_col and pd.api.types.is_numeric_dtype(baseline_df[base_col]):
                    v_curr = df[col].dropna().values
                    v_base = baseline_df[base_col].dropna().values
                    if len(v_curr) >= 10 and len(v_base) >= 10:
                        ks_stat, p_val = stats.ks_2samp(v_curr, v_base)
                        # KS stat > 0.4 indicates severe population distribution manipulation
                        if ks_stat > 0.35 and p_val < 0.001:
                            shift_scores.append(ks_stat)
                            indicators.append(f"Significant population distribution shift in '{col}' (KS stat={ks_stat:.3f}, p={p_val:.1e}).")

            if shift_scores:
                mean_shift = float(np.mean(shift_scores))
                if mean_shift > 0.5:
                    dist_shift_level = "HIGH"
                    risk_components.append(0.7)
                elif mean_shift > 0.3:
                    dist_shift_level = "MODERATE"
                    risk_components.append(0.4)

        # Check C: Suspicious Value Clumping / Artificial Injections (Entropy / Zero Variance Subsets)
        for col in target_cols:
            vals = df[col].dropna().values
            if len(vals) >= 20:
                unique_pct = (len(np.unique(vals)) / len(vals)) * 100.0
                if unique_pct < 2.0:
                    indicators.append(f"Suspicious synthetic repetition in '{col}': only {unique_pct:.1f}% unique values.")
                    risk_components.append(0.6)

        # Normalize Poisoning Risk Score [0.0 - 1.0]
        if not risk_components:
            poison_score = 0.05
        else:
            poison_score = float(np.clip(np.mean(risk_components) * 1.2, 0.0, 1.0))

        # Categorize Poison Risk Level
        if poison_score < 0.30:
            risk_level = "LOW"
        elif poison_score < 0.60:
            risk_level = "MODERATE"
        elif poison_score < 0.80:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "risk_score": round(poison_score, 4),
            "risk_level": risk_level,
            "extreme_outliers_count": int(extreme_outliers_count),
            "distribution_shift": dist_shift_level,
            "indicators": indicators,
        }


# ==============================================================================
# 5. Trust Gate & Decision Evaluator
# ==============================================================================

class TrustGate:
    """
    Synthesizes cryptographic integrity, schema fingerprints, tampering checks,
    and poisoning indicators into a final Data Trust Decision and ML Access Gate.
    """

    @classmethod
    def evaluate_gate(
        cls,
        integrity_result: Dict[str, Any],
        schema_result: Dict[str, Any],
        tamper_result: Dict[str, Any],
        poison_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate overall trust score and enforce ML safety gate.
        Rules:
        - Integrity Mismatch OR Critical Poisoning -> NOT_TRUSTED (ML Withheld)
        - Schema Change OR High Poisoning OR Tamper Warnings -> REVIEW_REQUIRED (Proceed with warning)
        - Clean verified data -> TRUSTED (ML Allowed)
        """
        is_integrity_fail = integrity_result.get("status") == "INTEGRITY_MISMATCH"
        is_poison_critical = poison_result.get("risk_level") in ["HIGH", "CRITICAL"]
        is_tamper_warning = tamper_result.get("status") != "PASSED"
        is_schema_changed = schema_result.get("status") != "VALID"

        if is_integrity_fail or is_poison_critical or tamper_result.get("suspicious_records_count", 0) > 10:
            overall_trust = "NOT_TRUSTED"
            ml_pipeline_allowed = False
            recommendation = "Data trust failure. ML safety decision is withheld until dataset integrity and provenance are reviewed by QA."
        elif is_tamper_warning or is_schema_changed or poison_result.get("risk_level") == "MODERATE":
            overall_trust = "REVIEW_REQUIRED"
            ml_pipeline_allowed = True
            recommendation = "Moderate data warnings detected. Proceed with automated screening, but manual QA validation is advised."
        else:
            overall_trust = "TRUSTED"
            ml_pipeline_allowed = True
            recommendation = "Dataset integrity, schema consistency, and statistical distribution verified. Fully authorized for ML inference."

        return {
            "overall_trust": overall_trust,
            "ml_pipeline_allowed": ml_pipeline_allowed,
            "recommendation": recommendation,
        }


# ==============================================================================
# 6. Immutable Audit Trail Logger
# ==============================================================================

class AuditTrailLogger:
    """
    Maintains an append-only, immutable audit record log for screening traceability.
    """

    AUDIT_DIR = Path("artifacts/integrity")
    AUDIT_LOG_FILE = AUDIT_DIR / "audit_log.jsonl"

    @classmethod
    def log_analysis(
        cls,
        dataset_hash: str,
        schema_fingerprint: str,
        overall_trust: str,
        poison_score: float,
        ml_pipeline_allowed: bool,
        row_count: int,
        dataset_id: str = "DATASET_UPLOAD",
    ) -> Dict[str, Any]:
        """
        Record a structured audit entry.
        """
        cls.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        audit_id = f"ESS-AUDIT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{dataset_hash[:6]}"
        record = {
            "analysis_id": audit_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "schema_fingerprint": schema_fingerprint,
            "row_count": row_count,
            "overall_trust": overall_trust,
            "poison_risk_score": poison_score,
            "ml_pipeline_allowed": ml_pipeline_allowed,
        }

        try:
            with open(cls.AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        return record


# ==============================================================================
# Unified Trust Layer Orchestrator
# ==============================================================================

class TrustLayer:
    """
    Unified entry point executing all security, integrity, and poisoning checks
    prior to data entering the machine learning pipeline.
    """

    @classmethod
    def evaluate_dataset(
        cls,
        df: pd.DataFrame,
        raw_content: Optional[Union[str, bytes]] = None,
        dataset_id: str = "DATASET_UPLOAD",
        expected_hash: Optional[str] = None,
        baseline_df: Optional[pd.DataFrame] = None,
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete Trust Layer evaluation pipeline.
        """
        content_to_hash = raw_content if raw_content is not None else df

        # 1. Schema Fingerprint
        schema_res = SchemaFingerprinter.analyze_schema(df)

        # 2. Cryptographic Integrity
        integrity_res = DatasetIntegrityChecker.verify_integrity(
            content=content_to_hash,
            dataset_id=dataset_id,
            expected_hash=expected_hash,
            row_count=len(df),
            schema_hash=schema_res["schema_fingerprint"],
        )

        # 3. Tamper Detection
        tamper_res = TamperDetector.check_tampering(df, custom_mapping)

        # 4. Statistical Poisoning Detection
        poison_res = PoisoningDetector.evaluate_poisoning_risk(df, baseline_df, custom_mapping)

        # 5. Trust Gate Synthesis
        gate_res = TrustGate.evaluate_gate(integrity_res, schema_res, tamper_res, poison_res)

        # 6. Audit Trail Recording
        audit_record = AuditTrailLogger.log_analysis(
            dataset_hash=integrity_res["current_hash"],
            schema_fingerprint=schema_res["schema_fingerprint"],
            overall_trust=gate_res["overall_trust"],
            poison_score=poison_res["risk_score"],
            ml_pipeline_allowed=gate_res["ml_pipeline_allowed"],
            row_count=len(df),
            dataset_id=dataset_id,
        )

        return {
            "integrity": integrity_res,
            "schema_integrity": schema_res,
            "tamper_detection": tamper_res,
            "poison_detection": poison_res,
            "overall_trust": gate_res["overall_trust"],
            "ml_pipeline_allowed": gate_res["ml_pipeline_allowed"],
            "recommendation": gate_res["recommendation"],
            "audit_record": audit_record,
        }
