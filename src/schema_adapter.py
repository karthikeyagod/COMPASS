"""
Generic ESS Dataset Profiler & Schema Adapter for SIH 260170.

Enables the existing ML screening platform to ingest arbitrary ESS CSV datasets
with different column naming conventions, time-series intervals, and parameter types.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class DatasetProfiler:
    """
    Lightweight, deterministic dataset inspector and column pattern detector.
    Identifies component IDs, manufacturing lots, time-series measurement intervals,
    secondary sensors, and parameter types without external dependencies or LLMs.
    """

    # Heuristic regex patterns for common engineering column conventions
    PATTERNS = {
        "component_id": [
            r"^component[_\-\s]?id$",
            r"^device[_\-\s]?id$",
            r"^ic[_\-\s]?id$",
            r"^part[_\-\s]?id$",
            r"^unit[_\-\s]?id$",
            r"^serial[_\-\s]?(?:no|num|number)?$",
            r"^dut[_\-\s]?id$",
            r"^chip[_\-\s]?id$",
            r"^sample[_\-\s]?id$",
            r"^id$",
        ],
        "lot_id": [
            r"^lot[_\-\s]?id$",
            r"^batch[_\-\s]?id$",
            r"^manufacturing[_\-\s]?lot$",
            r"^wafer[_\-\s]?lot$",
            r"^batch$",
            r"^lot$",
            r"^fab[_\-\s]?lot$",
        ],
        "value_0h": [
            r"^value[_\-\s]?0h(?:[_\-\s]?[uµm]?a)?$",
            r"^leakage[_\-\s]?0h(?:[_\-\s]?[uµm]?a)?$",
            r"^leak[_\-\s]?0(?:h)?$",
            r"^i[_\-\s]?0h?$",
            r"^val[_\-\s]?0h?$",
            r"^0h$",
            r"^t0$",
            r"^baseline[_\-\s]?(?:leakage|val|value)?$",
        ],
        "value_24h": [
            r"^value[_\-\s]?24h(?:[_\-\s]?[uµm]?a)?$",
            r"^leakage[_\-\s]?24h(?:[_\-\s]?[uµm]?a)?$",
            r"^leak[_\-\s]?24(?:h)?$",
            r"^i[_\-\s]?24h?$",
            r"^val[_\-\s]?24h?$",
            r"^24h$",
            r"^t24$",
        ],
        "value_96h": [
            r"^value[_\-\s]?96h(?:[_\-\s]?[uµm]?a)?$",
            r"^leakage[_\-\s]?96h(?:[_\-\s]?[uµm]?a)?$",
            r"^leak[_\-\s]?96(?:h)?$",
            r"^i[_\-\s]?96h?$",
            r"^val[_\-\s]?96h?$",
            r"^96h$",
            r"^t96$",
        ],
        "value_168h": [
            r"^value[_\-\s]?168h(?:[_\-\s]?[uµm]?a)?$",
            r"^leakage[_\-\s]?168h(?:[_\-\s]?[uµm]?a)?$",
            r"^leak[_\-\s]?168(?:h)?$",
            r"^i[_\-\s]?168h?$",
            r"^val[_\-\s]?168h?$",
            r"^168h$",
            r"^t168$",
            r"^final[_\-\s]?(?:leakage|val|value)?$",
        ],
        "delay_0h": [
            r"^propagationdelay[_\-\s]?0h(?:[_\-\s]?ns)?$",
            r"^delay[_\-\s]?0h?$",
            r"^tpd[_\-\s]?0h?$",
        ],
        "delay_24h": [
            r"^propagationdelay[_\-\s]?24h(?:[_\-\s]?ns)?$",
            r"^delay[_\-\s]?24h?$",
            r"^tpd[_\-\s]?24h?$",
        ],
        "iddq_0h": [
            r"^iddq[_\-\s]?0h(?:[_\-\s]?ma)?$",
            r"^iddq[_\-\s]?0$",
            r"^isupply[_\-\s]?0h?$",
        ],
        "iddq_24h": [
            r"^iddq[_\-\s]?24h(?:[_\-\s]?ma)?$",
            r"^iddq[_\-\s]?24$",
            r"^isupply[_\-\s]?24h?$",
        ],
        "datasheet_limit": [
            r"^datasheet[_\-\s]?(?:leakage[_\-\s]?)?limit(?:[_\-\s]?[uµm]?a)?$",
            r"^limit[_\-\s]?(?:ua|u)?$",
            r"^spec[_\-\s]?limit$",
            r"^upper[_\-\s]?limit$",
            r"^max[_\-\s]?limit$",
        ],
        "temperature": [
            r"^temperature(?:[_\-\s]?c)?$",
            r"^temp(?:[_\-\s]?c)?$",
            r"^chamber[_\-\s]?temp$",
        ],
        "latent_defect": [
            r"^latent[_\-\s]?defect$",
            r"^defect$",
            r"^is[_\-\s]?defect$",
            r"^ground[_\-\s]?truth$",
            r"^label$",
            r"^failed$",
        ],
    }

    @classmethod
    def match_column(cls, col_name: str, pattern_key: str) -> bool:
        """Test whether a column name matches any regex for a given target semantic key."""
        clean_name = str(col_name).strip().lower()
        patterns = cls.PATTERNS.get(pattern_key, [])
        for pat in patterns:
            if re.match(pat, clean_name):
                return True
        return False

    @classmethod
    def profile_dataframe(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Inspect an uploaded dataframe and return a structured profile and column mapping.
        """
        if df is None or len(df) == 0:
            return {
                "is_compatible": False,
                "error": "Uploaded CSV dataset is empty.",
                "components_count": 0,
                "lots_count": 0,
                "rows_count": 0,
                "columns_count": 0,
            }

        columns = list(df.columns)
        mapping: Dict[str, Optional[str]] = {key: None for key in cls.PATTERNS.keys()}
        confidence: Dict[str, float] = {key: 0.0 for key in cls.PATTERNS.keys()}

        # 1. Match columns against pattern heuristics
        for col in columns:
            for key in cls.PATTERNS.keys():
                if mapping[key] is None and cls.match_column(col, key):
                    mapping[key] = col
                    confidence[key] = 1.0

        # 2. Extract detected summary metrics
        cid_col = mapping["component_id"]
        lot_col = mapping["lot_id"]
        val0_col = mapping["value_0h"]
        val24_col = mapping["value_24h"]
        val96_col = mapping["value_96h"]
        val168_col = mapping["value_168h"]

        n_components = int(df[cid_col].nunique()) if cid_col and cid_col in df.columns else len(df)
        n_lots = int(df[lot_col].nunique()) if lot_col and lot_col in df.columns else (1 if lot_col else 0)

        # Detect primary parameter type
        detected_parameter = "Not detected"
        if any(mapping[k] for k in ["value_0h", "value_24h", "value_96h", "value_168h"]):
            sample_col = mapping["value_0h"] or mapping["value_24h"] or mapping["value_168h"]
            if sample_col and any(t in sample_col.lower() for t in ["leak", "value", "i_", "curr"]):
                detected_parameter = "Leakage Current (µA)"
            elif sample_col and "delay" in sample_col.lower():
                detected_parameter = "Propagation Delay (ns)"
            else:
                detected_parameter = "Parametric Drift"

        # Time points availability
        time_points = {
            "0h": bool(val0_col and val0_col in df.columns),
            "24h": bool(val24_col and val24_col in df.columns),
            "96h": bool(val96_col and val96_col in df.columns),
            "168h": bool(val168_col and val168_col in df.columns),
        }

        # Temperature detection
        temp_val = "Not detected"
        if mapping["temperature"] and mapping["temperature"] in df.columns:
            try:
                median_temp = float(df[mapping["temperature"]].median())
                temp_val = f"{median_temp:.0f}°C"
            except Exception:
                temp_val = "125°C"
        elif any(time_points.values()):
            temp_val = "125°C (Nominal Burn-in)"

        # Datasheet limit detection
        limit_val = "Not detected"
        if mapping["datasheet_limit"] and mapping["datasheet_limit"] in df.columns:
            try:
                mean_lim = float(df[mapping["datasheet_limit"]].mean())
                limit_val = f"{mean_lim:.1f} µA"
            except Exception:
                limit_val = "50.0 µA"
        elif time_points["0h"]:
            limit_val = "50.0 µA (Standard)"

        # Validation Checks & Warnings
        validation_errors = []
        validation_warnings = []

        if not cid_col:
            validation_errors.append("Required component identifier column was not detected.")
        if not (time_points["0h"] or time_points["24h"]):
            validation_errors.append("No early time-series measurements (0h or 24h) were detected.")

        if val0_col and val0_col in df.columns:
            n_miss_0h = int(df[val0_col].isna().sum())
            if n_miss_0h > 0:
                validation_warnings.append(f"{n_miss_0h} missing 0h measurements detected (will be imputed).")

        if val24_col and val24_col in df.columns:
            n_miss_24h = int(df[val24_col].isna().sum())
            if n_miss_24h > 0:
                validation_warnings.append(f"{n_miss_24h} missing 24h measurements detected (will be imputed).")

        if not time_points["96h"]:
            validation_warnings.append("96h mid-cycle measurement not detected (optional).")

        if not time_points["168h"]:
            validation_warnings.append("168h qualification measurement not detected (retrospective).")

        if cid_col and cid_col in df.columns:
            n_dup = int(df[cid_col].duplicated().sum())
            if n_dup > 0:
                validation_warnings.append(f"{n_dup} duplicate component identifiers detected.")

        # Compatibility Assessment
        is_compatible = len(validation_errors) == 0
        compatibility = {
            "module_a": bool(time_points["0h"] and time_points["24h"]),
            "module_b": bool(time_points["0h"] and time_points["24h"]),
            "safety_slope": bool(time_points["0h"] and time_points["24h"]),
            "traditional_baseline": bool(time_points["168h"]),
            "reasons": [],
        }

        if not compatibility["module_a"]:
            compatibility["reasons"].append("Module A & B require at least 0h and 24h measurements.")
        if not compatibility["traditional_baseline"]:
            compatibility["reasons"].append("Traditional static limit evaluation requires 168h measurement.")

        return {
            "is_compatible": is_compatible,
            "components_count": n_components,
            "lots_count": n_lots if n_lots > 0 else 1,
            "rows_count": len(df),
            "columns_count": len(columns),
            "columns_available": columns,
            "detected_parameter": detected_parameter,
            "time_points_detected": time_points,
            "temperature_detected": temp_val,
            "static_limit_detected": limit_val,
            "suggested_mapping": mapping,
            "mapping_confidence": confidence,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "compatibility": compatibility,
        }


class DatasetAdapter:
    """
    Adapter layer normalizing any arbitrary uploaded ESS dataframe to the
    internal canonical schema expected by FeatureEngineer and InferencePipeline.
    """

    CANONICAL_COLUMNS = [
        "Component_ID",
        "Lot_ID",
        "Value_0h_uA",
        "Value_24h_uA",
        "Value_96h_uA",
        "Value_168h_uA",
        "PropagationDelay_0h_ns",
        "PropagationDelay_24h_ns",
        "Iddq_0h_mA",
        "Iddq_24h_mA",
        "Datasheet_Leakage_Limit_uA",
        "Temperature_C",
        "Latent_Defect",
    ]

    @classmethod
    def normalize_dataframe(
        cls,
        df: pd.DataFrame,
        custom_mapping: Optional[Dict[str, Optional[str]]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Transform raw uploaded DataFrame into canonical schema using detected or custom mapping.

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Normalized DataFrame conforming to internal schema and list of transformation notes.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(), ["Input dataframe is empty."]

        notes: List[str] = []
        df_raw = df.copy()

        # Step 1: Resolve Mapping
        if not custom_mapping:
            profile = DatasetProfiler.profile_dataframe(df_raw)
            mapping = profile["suggested_mapping"]
        else:
            mapping = custom_mapping

        normalized = pd.DataFrame()

        # Component ID
        cid_col = mapping.get("component_id")
        if cid_col and cid_col in df_raw.columns:
            normalized["Component_ID"] = df_raw[cid_col].astype(str)
        else:
            normalized["Component_ID"] = [f"IC_GEN_{i+1:05d}" for i in range(len(df_raw))]
            notes.append("Generated generic Component_IDs (e.g. IC_GEN_00001).")

        # Lot ID
        lot_col = mapping.get("lot_id")
        if lot_col and lot_col in df_raw.columns:
            normalized["Lot_ID"] = df_raw[lot_col].astype(str)
        else:
            normalized["Lot_ID"] = "LOT_GENERIC"
            notes.append("Assigned default Lot_ID 'LOT_GENERIC'.")

        # Value 0h
        v0_col = mapping.get("value_0h")
        if v0_col and v0_col in df_raw.columns:
            normalized["Value_0h_uA"] = pd.to_numeric(df_raw[v0_col], errors="coerce").fillna(10.0)
        else:
            normalized["Value_0h_uA"] = 10.0
            notes.append("Value_0h_uA not provided; initialized to nominal 10.0 uA.")

        # Value 24h
        v24_col = mapping.get("value_24h")
        if v24_col and v24_col in df_raw.columns:
            normalized["Value_24h_uA"] = pd.to_numeric(df_raw[v24_col], errors="coerce")
        else:
            normalized["Value_24h_uA"] = np.nan
            notes.append("Value_24h_uA not provided (Early 24h screening input missing).")

        # Value 96h (Ground Truth / Post-Burn-in Evaluation Data)
        v96_col = mapping.get("value_96h")
        if v96_col and v96_col in df_raw.columns:
            normalized["Value_96h_uA"] = pd.to_numeric(df_raw[v96_col], errors="coerce")
        else:
            normalized["Value_96h_uA"] = np.nan

        # Value 168h (Ground Truth / Post-Burn-in Evaluation Data)
        v168_col = mapping.get("value_168h")
        if v168_col and v168_col in df_raw.columns:
            normalized["Value_168h_uA"] = pd.to_numeric(df_raw[v168_col], errors="coerce")
        else:
            normalized["Value_168h_uA"] = np.nan

        # Propagation Delays
        d0_col = mapping.get("delay_0h")
        if d0_col and d0_col in df_raw.columns:
            normalized["PropagationDelay_0h_ns"] = pd.to_numeric(df_raw[d0_col], errors="coerce").fillna(2.015)
        else:
            normalized["PropagationDelay_0h_ns"] = 2.015

        d24_col = mapping.get("delay_24h")
        if d24_col and d24_col in df_raw.columns:
            normalized["PropagationDelay_24h_ns"] = pd.to_numeric(df_raw[d24_col], errors="coerce").fillna(2.218)
        else:
            delta_leak = normalized["Value_24h_uA"] - normalized["Value_0h_uA"]
            normalized["PropagationDelay_24h_ns"] = normalized["PropagationDelay_0h_ns"] + np.maximum(0.0, delta_leak * 0.1)

        # Iddq
        i0_col = mapping.get("iddq_0h")
        if i0_col and i0_col in df_raw.columns:
            normalized["Iddq_0h_mA"] = pd.to_numeric(df_raw[i0_col], errors="coerce").fillna(1.20)
        else:
            normalized["Iddq_0h_mA"] = 1.20

        i24_col = mapping.get("iddq_24h")
        if i24_col and i24_col in df_raw.columns:
            normalized["Iddq_24h_mA"] = pd.to_numeric(df_raw[i24_col], errors="coerce").fillna(1.35)
        else:
            delta_leak = normalized["Value_24h_uA"] - normalized["Value_0h_uA"]
            normalized["Iddq_24h_mA"] = normalized["Iddq_0h_mA"] + np.maximum(0.0, delta_leak * 0.05)

        # Datasheet Limit
        lim_col = mapping.get("datasheet_limit")
        if lim_col and lim_col in df_raw.columns:
            normalized["Datasheet_Leakage_Limit_uA"] = pd.to_numeric(df_raw[lim_col], errors="coerce").fillna(50.0)
        else:
            normalized["Datasheet_Leakage_Limit_uA"] = 50.0

        # Temperature
        temp_col = mapping.get("temperature")
        if temp_col and temp_col in df_raw.columns:
            normalized["Temperature_C"] = pd.to_numeric(df_raw[temp_col], errors="coerce").fillna(125.0)
        else:
            normalized["Temperature_C"] = 125.0

        # Ground Truth Label (if present)
        gt_col = mapping.get("latent_defect")
        if gt_col and gt_col in df_raw.columns:
            normalized["Latent_Defect"] = pd.to_numeric(df_raw[gt_col], errors="coerce").fillna(0).astype(int)
        else:
            normalized["Latent_Defect"] = 0

        return normalized, notes
