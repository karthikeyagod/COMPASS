"""
Tests for EDA and Dataset Inspection module (Phase 1).
"""

from pathlib import Path
import pandas as pd
import pytest
from src.eda_inspect import inspect_dataset


def test_inspect_dataset_structure():
    """Verify that inspect_dataset properly computes all required structural metrics."""
    data_path = Path("data/sih_260170_synthetic_ess_dataset.csv")
    assert data_path.exists(), "Dataset file must exist under data/"
    
    df = pd.read_csv(data_path)
    inspection = inspect_dataset(df)

    assert inspection["num_rows"] == 12000
    assert inspection["num_cols"] == 19
    assert inspection["duplicate_records"] == 0
    assert inspection["unique_components"] == 12000
    assert inspection["unique_lots"] == 60
    assert sum(inspection["missing_values"].values()) == 0


def test_sih_latent_escape_rate():
    """Verify that the core SIH latent escape calculation accurately identifies the escape rate."""
    data_path = Path("data/sih_260170_synthetic_ess_dataset.csv")
    df = pd.read_csv(data_path)
    inspection = inspect_dataset(df)
    
    sih = inspection["sih_analysis"]
    assert sih["total_latent_defects"] == 1024
    assert sih["latent_defects_passing_limit"] == 1024
    assert sih["latent_escape_rate_percent"] == 100.0
