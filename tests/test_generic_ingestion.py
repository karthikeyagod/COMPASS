"""
Unit & Integration Tests for Generic ESS Dataset Ingestion, Profiler, and Adapter.
Validates arbitrary CSV schema adaptation, missing channel handling, compatibility checks, and API routes.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.schema_adapter import DatasetAdapter, DatasetProfiler


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_profiler_standard_sih_dataset():
    """Test profiler on standard SIH dataset format."""
    df = pd.DataFrame({
        "Component_ID": ["IC_001", "IC_002"],
        "Lot_ID": ["LOT_01", "LOT_01"],
        "Value_0h_uA": [10.1, 10.2],
        "Value_24h_uA": [11.5, 10.4],
        "Value_96h_uA": [18.2, 11.1],
        "Value_168h_uA": [33.4, 12.0],
        "Datasheet_Leakage_Limit_uA": [50.0, 50.0],
    })

    profile = DatasetProfiler.profile_dataframe(df)
    assert profile["is_compatible"] is True
    assert profile["components_count"] == 2
    assert profile["lots_count"] == 1
    assert profile["time_points_detected"]["0h"] is True
    assert profile["time_points_detected"]["24h"] is True
    assert profile["time_points_detected"]["96h"] is True
    assert profile["time_points_detected"]["168h"] is True
    assert profile["compatibility"]["module_a"] is True
    assert profile["compatibility"]["module_b"] is True


def test_profiler_alternative_column_names():
    """Test profiler on alternative naming conventions (Device_ID, Batch, Leak_0, Leak_24)."""
    df = pd.DataFrame({
        "Device_ID": ["DEV_001", "DEV_002", "DEV_003"],
        "Batch": ["BATCH_A", "BATCH_A", "BATCH_B"],
        "Leak_0": [10.0, 10.2, 10.1],
        "Leak_24": [11.6, 10.4, 11.8],
        "Leak_96": [18.5, 11.2, 19.0],
        "Leak_168": [34.0, 12.1, 35.5],
        "Limit": [50.0, 50.0, 50.0],
    })

    profile = DatasetProfiler.profile_dataframe(df)
    assert profile["is_compatible"] is True
    assert profile["components_count"] == 3
    assert profile["lots_count"] == 2
    assert profile["suggested_mapping"]["component_id"] == "Device_ID"
    assert profile["suggested_mapping"]["lot_id"] == "Batch"
    assert profile["suggested_mapping"]["value_0h"] == "Leak_0"
    assert profile["suggested_mapping"]["value_24h"] == "Leak_24"


def test_adapter_normalization():
    """Test adapter normalizes alternative columns into canonical internal schema."""
    df_raw = pd.DataFrame({
        "Device_ID": ["DEV_100", "DEV_101"],
        "Batch": ["LOT_X", "LOT_X"],
        "Leak_0": [10.12, 10.05],
        "Leak_24": [11.56, 10.32],
    })

    normalized, notes = DatasetAdapter.normalize_dataframe(df_raw)
    assert "Component_ID" in normalized.columns
    assert "Lot_ID" in normalized.columns
    assert "Value_0h_uA" in normalized.columns
    assert "Value_24h_uA" in normalized.columns
    assert normalized["Component_ID"].iloc[0] == "DEV_100"
    assert normalized["Lot_ID"].iloc[0] == "LOT_X"
    assert normalized["Value_0h_uA"].iloc[0] == pytest.approx(10.12)
    assert normalized["Value_24h_uA"].iloc[0] == pytest.approx(11.56)
    assert "PropagationDelay_0h_ns" in normalized.columns


def test_missing_96h_graceful_handling():
    """Test dataset missing 96h measurement runs cleanly without error."""
    df = pd.DataFrame({
        "Component_ID": ["IC_M1", "IC_M2"],
        "Lot_ID": ["LOT_M", "LOT_M"],
        "Value_0h_uA": [10.0, 10.1],
        "Value_24h_uA": [10.3, 11.5],
        "Value_168h_uA": [12.0, 33.0],
    })

    profile = DatasetProfiler.profile_dataframe(df)
    assert profile["time_points_detected"]["96h"] is False
    assert profile["compatibility"]["module_a"] is True
    assert profile["compatibility"]["module_b"] is True

    normalized, notes = DatasetAdapter.normalize_dataframe(df)
    assert "Value_96h_uA" in normalized.columns
    assert normalized["Value_96h_uA"].isna().all()


def test_missing_24h_model_compatibility():
    """Test dataset missing 24h flag is detected and Module B is marked unavailable."""
    df = pd.DataFrame({
        "Component_ID": ["IC_NO24_1", "IC_NO24_2"],
        "Lot_ID": ["LOT_N", "LOT_N"],
        "Value_0h_uA": [10.0, 10.1],
        "Value_168h_uA": [12.0, 33.0],
    })

    profile = DatasetProfiler.profile_dataframe(df)
    assert profile["time_points_detected"]["24h"] is False
    assert profile["compatibility"]["module_b"] is False
    assert any("require at least 0h and 24h" in r for r in profile["compatibility"]["reasons"])


def test_incompatible_dataset_handling():
    """Test completely unrelated CSV (e.g. sales.csv) is flagged as incompatible."""
    df_sales = pd.DataFrame({
        "Order_ID": [1001, 1002, 1003],
        "Customer_Name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
        "Revenue_USD": [5000.0, 12000.0, 8500.0],
        "Quantity": [10, 25, 15],
    })

    profile = DatasetProfiler.profile_dataframe(df_sales)
    assert profile["is_compatible"] is False
    assert len(profile["validation_errors"]) > 0


def test_api_dataset_profile_endpoint(client):
    """Test POST /dataset/profile endpoint with raw CSV string."""
    csv_text = "Device_ID,Batch,Leak_0,Leak_24\nDEV_1,LOT_A,10.1,11.5\nDEV_2,LOT_A,10.0,10.3\n"
    response = client.post("/dataset/profile", json={"csv_text": csv_text})
    assert response.status_code == 200
    data = response.json()
    assert data["is_compatible"] is True
    assert data["components_count"] == 2
    assert data["suggested_mapping"]["component_id"] == "Device_ID"


def test_api_dataset_analyze_endpoint(client):
    """Test POST /dataset/analyze endpoint executes full ML pipeline on uploaded CSV."""
    csv_text = "Device_ID,Batch,Leak_0,Leak_24,Leak_168\nDEV_1,LOT_A,10.12,11.56,33.40\nDEV_2,LOT_A,10.05,10.32,12.10\n"
    response = client.post("/dataset/analyze", json={"csv_text": csv_text})
    assert response.status_code == 200
    data = response.json()
    assert data["total_components"] == 2
    assert "summary" in data
    assert "records" in data
    assert len(data["records"]) == 2
    assert "Predicted_168h" in data["records"][0]
    assert "Risk_Level" in data["records"][0]
