# COMPASS: Component Anomaly Screening and Surveillance
### SIH Problem Statement 260170: Latent Defect Detection in Electronic Components during ESS

[![Python 3.11](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/pytest-56%2F56%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end intelligent machine learning framework for **early latent defect detection, 168-hour parametric drift prognostics, degradation safety slope compliance, and root-cause explainability** for semiconductor components undergoing Environmental Stress Screening (ESS).

---

## 1. Problem Overview & The Latent Defect Dilemma

During the manufacturing and qualification of mission-critical electronics (aerospace, defense, and automotive), components undergo **Environmental Stress Screening (ESS)** (e.g., $125^\circ\text{C}$ thermal burn-in over $168\,\text{hours}$).

### Why Static Datasheet Limits Fail:
Traditional qualification relies on **static threshold verification** ($\text{Value}_{168\text{h}} \le 50.0\,\mu\text{A} \implies \text{PASS}$).
* **Sub-Threshold Degradation:** Latent defects (gate-oxide micro-voids, dielectric breakdown, electromigration) degrade along non-linear degradation paths but remain below $50.0\,\mu\text{A}$ at 168 hours (max observed defect: $44.20\,\mu\text{A}$).
* **100% Silent Escape Rate:** Traditional static screening marks all latent defects as conforming, allowing compromised components to be shipped to mission-critical systems where they fail catastrophically.
* **Excessive Burn-in Cost:** Traditional screening consumes the entire $168\,\text{hours}$ ($7\,\text{days}$) of burn-in chamber time.

### The AI Solution:
This framework evaluates early dynamic kinetics at **$24\,\text{hours}$** ($85.71\%$ faster), intercepting **$100.0\%$ of latent defect escapes** while maintaining zero false alarms on conforming components.

---

## 2. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        QA ENGINEER DASHBOARD                           │
│  Command Center KPIs  │  Batch Analytics  │  Data Grid  │  Chart.js    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST / JSON
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI REST BACKEND                              │
│  GET /health  │  POST /predict/component  │  GET /components           │
│  GET /batch/{lot}  │  GET /component/{id} │  GET /model/metrics        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  END-TO-END INFERENCE PIPELINE                         │
│  Quality Validation Layer -> Stage-Aware Feature Extraction (t <= 24h) │
└───────────────────┬───────────────────────────────┬────────────────────┘
                    │                               │
                    ▼                               ▼
    ┌──────────────────────────────┐ ┌──────────────────────────────┐
    │  MODULE A (Isolation Forest) │ │  MODULE B (Gradient Boost)   │
    │  Kinetic Anomaly Detection   │ │  168h Drift Continuous Prog. │
    └───────────────┬──────────────┘ └──────────────┬───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  SAFETY SLOPE ENVELOPE (Tukey Robust Boundary: 0.03707 uA/h)  │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  COMPONENT RISK ENGINE (Composite 0-100 & Decision Matrix)    │
    └───────────────────────────────┬───────────────────────────────┘
                                    ▼
    ┌───────────────────────────────────────────────────────────────┐
    │  EXPLAINABLE AI (TreeExplainer SHAP + 4-Point QA Diagnostics) │
    └───────────────────────────────────────────────────────────────┘
```

---

## 3. Machine Learning Subsystems

1. **Stage-Aware Feature Engineering (`src/feature_engineering.py`):**
   - Extracts kinetic drift velocity ($dI/dt$), percentage change, gate delay shifts ($\Delta t_{\text{pd}}$), and quiescent supply current shifts ($\Delta I_{\text{ddq}}$).
   - Computes intra-lot robust Z-scores using median and IQR.
   - Enforces data leakage prevention (`STAGE_FEATURE_REGISTRY`).

2. **Module A — Dynamic Anomaly Detection (`src/anomaly_detection.py`):**
   - Isolation Forest model ($N=150$ estimators) trained on 18 early dynamic features.
   - Calibrated safety threshold ($\tau = 0.2365$) optimized for zero false-negative escapes.

3. **Module B — 168-Hour Parametric Drift Forecasting (`src/drift_prediction.py`):**
   - Gradient Boosting Regressor forecasting continuous 168h leakage from 24h inputs.
   - Achieves $\text{Test MAE} = 1.3216\,\mu\text{A}$ and $R^2 = 0.8625$ with physical monotonicity bounding.

4. **Safety Slope & Healthy Degradation Envelope (`src/safety_slope.py`):**
   - Derived non-parametric Tukey boundary ($\text{Safety\_Slope} = 0.037074\,\mu\text{A}/\text{hour}$).
   - Flags components whose degradation velocity breaches the healthy operational corridor.

5. **Component Risk Engine (`src/risk_engine.py`):**
   - Synthesizes anomaly score, drift forecast, safety slope violation, and lot deviation into a transparent $0-100$ Risk Score (`NORMAL`, `WATCH`, `HIGH_RISK`).

6. **Explainable AI (`src/explainability.py`):**
   - Combines `shap.TreeExplainer` local attributions with domain QA translation to generate 4-point plain-English root-cause justifications.

---

## 4. Benchmark Performance Results

Evaluation on held-out **Test Dataset** ($N = 1,770$ components across 9 unseen manufacturing lots):

| Performance Metric | Traditional Static ESS Screening | Dynamic AI Screening Platform | Operational Impact |
| :--- | :--- | :--- | :--- |
| **Screening Duration** | $168\,\text{hours}$ ($7.0\,\text{days}$) | **$24\,\text{hours}$ ($1.0\,\text{day}$)** | **$85.71\%$ Time Reduction** |
| **Latent Defect Recall** | **$0.00\%$** ($0 / 146$) | **$100.00\%$** ($146 / 146$) | **$100\%$ Defect Interception** |
| **False Negative Rate** | **$100.00\%$** ($146 / 146$ escapes) | **$0.00\%$** ($0 / 146$ escapes) | **Zero Silent Escapes** |
| **False Positives** | $0$ | **$0$** | **Zero False Alarms on Normals** |
| **Precision** | $0.00\%$ | **$100.00\%$** | Clean operational separation |
| **F1 Score** | $0.0000$ | **$1.0000$** | Optimal balance |
| **168h Forecast MAE** | N/A (Cannot predict) | **$1.3216\,\mu\text{A}$** | High kinetic fidelity |
| **168h Forecast $R^2$** | N/A | **$0.8625$** | Continuous prognostics |

---

## 5. Quickstart & Local Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.11.9)
- Virtual Environment tool (`venv`)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/your-org/sih-260170-ess-screening.git
cd sih-260170-ess-screening

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Launch the Application
```bash
# Start the FastAPI REST server & QA Dashboard
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

* **Interactive QA Dashboard:** `http://127.0.0.1:8000/dashboard` or `http://127.0.0.1:8000/`
* **Swagger API Documentation:** `http://127.0.0.1:8000/docs`
* **ReDoc API Documentation:** `http://127.0.0.1:8000/redoc`

---

## 6. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service readiness, model preloading status, and record count |
| `POST` | `/predict/component` | Primary screening endpoint (anomaly score, 168h forecast, risk score, XAI reasons) |
| `POST` | `/anomaly/analyze` | Dedicated Module A dynamic kinetic anomaly evaluation |
| `GET` | `/component/{id}` | Telemetry lookup and diagnostic evaluation for a component ID |
| `GET` | `/components` | Stream and filter components by Lot ID and search query |
| `GET` | `/batch/{lot_id}` | Manufacturing lot population analytics, defect rate, and sample ICs |
| `GET` | `/model/metrics` | Benchmark performance metrics across all screening modules |

---

## 7. QA Engineer Dashboard

The dashboard provides a high-contrast dark glassmorphic engineering interface:
1. **Command Center:** Real-time screening KPI counts (Total, Normal, Watch, High Risk, Static Failures, Dynamic Detections).
2. **Manufacturing Batch Analytics:** Population metrics, mean drift, median, and defect distribution for any manufacturing wafer lot.
3. **Component Screening Data Grid:** Searchable, filterable table with click-to-inspect interaction.
4. **Time-Series Trajectory Plot:** Interactive Chart.js graph displaying actual telemetry, predicted 168h trajectory, shaded healthy degradation envelope ($0.0371\,\mu\text{A}/\text{h}$), and static datasheet limit ($50.0\,\mu\text{A}$).
5. **Root-Cause Explanation Panel:** 10-second executive verdict, 4 diagnostic reasons, and physical attribution factors.
6. **Live Screening Sandbox:** Interactive tool to test custom $0\text{h}$ and $24\text{h}$ measurements live against the backend.

---

## 8. Training & Verification Scripts

```bash
# 1. Run full unit and integration test suite (56 tests)
pytest tests/

# 2. Re-run comprehensive Phase 15 evaluation & stress tests
python training/evaluate_final_system.py

# 3. Benchmark Module B Drift Regressors
python training/train_drift.py

# 4. Calibrate Safety Slope Envelope
python training/evaluate_safety_slope.py
```

---

## 9. Limitations & Future Work

### Limitations:
- **Synthetic Benchmark Dataset:** The dataset utilized for development and benchmarking is a calibrated synthetic dataset reflecting standard semiconductor Arrhenius degradation physics and intra-lot distributions. It is not raw classified flight telemetry.
- **Parametric Channels:** The current model suite is trained on leakage current, propagation delay, quiescent supply current, and temperature. Additional sensor channels (e.g., vibration PSD or acoustic emission) require training on corresponding multi-physics data.

### Future Work:
1. **Edge Firmware Deployment:** Porting inference models to C++/ONNX Runtime for microcontrollers embedded directly inside ATE test fixtures.
2. **Transfer Learning across Process Nodes:** Adapting the drift regressor across different silicon technology nodes ($28\,\text{nm} \to 7\,\text{nm}$) with minimal calibration data.
3. **Multi-Chamber Fleet Synchronization:** Connecting multiple screening facilities to a centralized model registry for continuous federated model updates.

---

## 10. Deliverables & Documentation Index

- [SIH Presentation & Demo Guide](docs/SIH_PRESENTATION_AND_DEMO_GUIDE.md)
- [Final Evaluation Report (Markdown)](outputs/final_evaluation_report.md)
- [Final Evaluation Results (JSON)](outputs/final_evaluation.json)
- [Multi-Phase Walkthrough Report](walkthrough.md)
- [Implementation Plan](implementation_plan.md)
