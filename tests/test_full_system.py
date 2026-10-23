"""
Full System Integration Test Suite (Phase 14).

Verifies end-to-end integration:
Frontend Routes -> FastAPI -> Inference Pipeline -> Module A -> Module B -> Safety Slope -> Risk Engine -> Explainability.
"""

from fastapi.testclient import TestClient
import pytest

from src.api import app
from src.inference import InferencePipeline, analyze_component


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Initialize FastAPI TestClient with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_full_system_workflow_latent_defect(client: TestClient):
    """
    Verify complete 8-step QA workflow on escaping latent defect:
    1. Select component
    2. View ESS measurements
    3. Run analysis
    4. Verify Anomaly score
    5. Verify Predicted 168h value
    6. Verify Safety slope result
    7. Verify Risk score
    8. Read explanation
    """
    # 1 & 2: Select component & retrieve ESS telemetry
    res_lookup = client.get("/component/IC_000006")
    assert res_lookup.status_code == 200
    comp_data = res_lookup.json()

    assert comp_data["component_id"] == "IC_000006"
    assert "supporting_measurements" in comp_data

    # 3: Run analysis on live endpoint
    payload = {
        "component_id": comp_data["component_id"],
        "lot_id": "LOT_001",
        "value_0h_uA": 10.12,
        "value_24h_uA": 11.56,
        "propagation_delay_0h_ns": 2.015,
        "propagation_delay_24h_ns": 2.218,
        "iddq_0h_mA": 1.20,
        "iddq_24h_mA": 1.35,
        "datasheet_limit_uA": 50.0,
    }
    res_eval = client.post("/predict/component", json=payload)
    assert res_eval.status_code == 200
    report = res_eval.json()

    # 4: Anomaly Score
    assert 0.0 <= report["anomaly_score"] <= 1.0
    assert report["anomaly_score"] > 0.50

    # 5: Predicted 168h Value (Datasheet Pass: True, < 50.0 uA)
    assert report["predicted_168h_uA"] < 50.0
    assert report["datasheet_pass"] is True

    # 6: Safety Slope Result (Breached)
    assert report["slope_exceeded"] == 1
    assert report["predicted_drift_rate_uA_per_h"] > report["safety_slope_uA_per_h"]

    # 7: Risk Score & Level
    assert report["risk_score"] >= 75.0
    assert report["risk_level"] == "HIGH_RISK"

    # 8: Explanation
    assert isinstance(report["explanation"], list)
    assert len(report["explanation"]) >= 3
    assert any("safety envelope" in r.lower() or "breaches" in r.lower() for r in report["explanation"])


def test_full_system_workflow_normal_component(client: TestClient):
    """
    Verify complete 8-step QA workflow on conforming normal component.
    """
    payload = {
        "component_id": "IC_NORMAL_STAGE14",
        "lot_id": "LOT_001",
        "value_0h_uA": 10.05,
        "value_24h_uA": 10.30,
        "propagation_delay_0h_ns": 2.010,
        "propagation_delay_24h_ns": 2.022,
        "iddq_0h_mA": 1.20,
        "iddq_24h_mA": 1.21,
        "datasheet_limit_uA": 50.0,
    }
    res = client.post("/predict/component", json=payload)
    assert res.status_code == 200
    report = res.json()

    assert report["risk_level"] == "NORMAL"
    assert report["risk_score"] < 35.0
    assert report["slope_exceeded"] == 0
    assert report["datasheet_pass"] is True
    assert len(report["explanation"]) > 0


def test_components_streaming_and_batch_integration(client: TestClient):
    """Verify live components listing and lot filtering."""
    # List all
    res_list = client.get("/components?limit=10&offset=0")
    assert res_list.status_code == 200
    data = res_list.json()
    assert data["total_records"] == 12000
    assert len(data["records"]) == 10

    # Filter by LOT_002
    res_lot = client.get("/components?lot_id=LOT_002&limit=5")
    assert res_lot.status_code == 200
    lot_data = res_lot.json()
    assert all(r["Lot_ID"] == "LOT_002" for r in lot_data["records"])


def test_invalid_input_and_error_handling(client: TestClient):
    """Verify robust error recovery on missing fields, unknown IDs, and out-of-range inputs."""
    # 404 on unknown component
    res_404 = client.get("/component/NON_EXISTENT_ID_99999")
    assert res_404.status_code == 404

    # 422 on missing required field
    res_422 = client.post("/predict/component", json={"component_id": "IC_MISSING"})
    assert res_422.status_code == 422

    # Extreme physical anomaly (sensor error handled gracefully)
    extreme_payload = {
        "component_id": "IC_EXTREME",
        "lot_id": "LOT_001",
        "value_0h_uA": -10.0,
        "value_24h_uA": 1200.0,
    }
    res_extreme = client.post("/predict/component", json=extreme_payload)
    assert res_extreme.status_code == 200
    ext_report = res_extreme.json()
    assert ext_report["risk_level"] == "HIGH_RISK"
    assert len(ext_report["validation_warnings"]) > 0
