"""
Unit and Integration tests for FastAPI REST Backend (Phase 12).
"""

from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from src.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Initialize FastAPI TestClient with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify /health endpoint returns service readiness and loaded models."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "models_loaded" in data
    assert data["models_loaded"]["anomaly_detector"] is True
    assert data["models_loaded"]["drift_predictor"] is True
    assert data["models_loaded"]["safety_slope_estimator"] is True
    assert data["models_loaded"]["risk_engine"] is True


def test_predict_component_normal(client: TestClient):
    """Verify /predict/component on normal component."""
    payload = {
        "component_id": "IC_API_NORMAL_01",
        "lot_id": "LOT_001",
        "value_0h_uA": 10.10,
        "value_24h_uA": 10.35,
        "propagation_delay_0h_ns": 2.01,
        "propagation_delay_24h_ns": 2.02,
        "iddq_0h_mA": 1.20,
        "iddq_24h_mA": 1.21,
        "datasheet_limit_uA": 50.0,
    }

    response = client.post("/predict/component", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["component_id"] == "IC_API_NORMAL_01"
    assert data["risk_level"] == "NORMAL"
    assert data["risk_score"] < 35.0
    assert data["datasheet_pass"] is True
    assert data["slope_exceeded"] == 0
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0


def test_predict_component_latent_defect(client: TestClient):
    """Verify /predict/component on escaping latent defect (Core SIH case)."""
    payload = {
        "component_id": "IC_004821",
        "lot_id": "LOT_012",
        "value_0h_uA": 10.20,
        "value_24h_uA": 17.40,
        "propagation_delay_0h_ns": 2.01,
        "propagation_delay_24h_ns": 2.45,
        "iddq_0h_mA": 1.20,
        "iddq_24h_mA": 1.80,
        "datasheet_limit_uA": 50.0,
    }

    response = client.post("/predict/component", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["component_id"] == "IC_004821"
    assert data["risk_level"] == "HIGH_RISK"
    assert data["risk_score"] >= 75.0
    assert data["anomaly_score"] > 0.50
    assert data["slope_exceeded"] == 1
    assert data["datasheet_pass"] is True  # Stays below 50 uA, caught by AI
    assert len(data["explanation"]) > 0


def test_anomaly_analyze_endpoint(client: TestClient):
    """Verify /anomaly/analyze dedicated Module A endpoint."""
    payload = {
        "component_id": "IC_ANO_001",
        "lot_id": "LOT_001",
        "value_0h_uA": 10.10,
        "value_24h_uA": 15.20,
    }

    response = client.post("/anomaly/analyze", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["component_id"] == "IC_ANO_001"
    assert data["anomaly_score"] > 0.0
    assert "is_anomaly" in data
    assert "early_drift_uA" in data
    assert "decision" in data


def test_get_component_by_id_success(client: TestClient):
    """Verify /component/{component_id} on valid component in dataset."""
    response = client.get("/component/IC_000001")
    assert response.status_code == 200

    data = response.json()
    assert data["component_id"] == "IC_000001"
    assert "risk_score" in data
    assert "predicted_168h_uA" in data


def test_get_component_by_id_not_found(client: TestClient):
    """Verify /component/{component_id} returns 404 on missing component."""
    response = client.get("/component/IC_INVALID_ID_999999")
    assert response.status_code == 404


def test_get_batch_by_lot_id_success(client: TestClient):
    """Verify /batch/{lot_id} on valid manufacturing lot."""
    response = client.get("/batch/LOT_001")
    assert response.status_code == 200

    data = response.json()
    assert data["lot_id"] == "LOT_001"
    assert data["total_components_in_lot"] > 0
    assert "lot_defect_rate_pct" in data
    assert "lot_mean_0h_leakage_uA" in data
    assert len(data["sample_component_ids"]) > 0


def test_get_batch_by_lot_id_not_found(client: TestClient):
    """Verify /batch/{lot_id} returns 404 on unknown lot."""
    response = client.get("/batch/LOT_NON_EXISTENT_999")
    assert response.status_code == 404


def test_get_model_metrics(client: TestClient):
    """Verify /model/metrics returns benchmark metrics across all modules."""
    response = client.get("/model/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "traditional_baseline" in data
    assert "module_a_anomaly_detection" in data
    assert "module_b_drift_prediction" in data
    assert "safety_slope_envelope" in data
    assert "component_risk_engine" in data

    assert data["traditional_baseline"]["recall"] == 0.0
    assert data["module_a_anomaly_detection"]["test_recall"] > 0.95
    assert data["module_b_drift_prediction"]["test_mae_uA"] < 2.0


def test_validation_error_on_bad_payload(client: TestClient):
    """Verify 422 Unprocessable Entity when required fields are missing or mistyped."""
    bad_payload = {
        "component_id": "IC_BAD",
        # Missing value_0h_uA and value_24h_uA
    }
    response = client.post("/predict/component", json=bad_payload)
    assert response.status_code == 422


def test_dashboard_frontend_serving(client: TestClient):
    """Verify dashboard HTML, CSS, and JS routes are properly served."""
    # Root dashboard route
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "SIH 260170" in res_root.text
    assert "Command Center" in res_root.text

    # /dashboard route
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "trajectoryChart" in res_dash.text

    # CSS stylesheet
    res_css = client.get("/index.css")
    assert res_css.status_code == 200
    assert "--bg-app" in res_css.text

    # JS application script
    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert "loadComponentsStream" in res_js.text

