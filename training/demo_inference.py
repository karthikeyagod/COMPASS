"""
Demo script for Phase 11 End-to-End Inference Pipeline across 6 Scenarios.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import analyze_component

scenarios = [
    ("Scenario 1: Normal Component", {
        "Component_ID": "IC_NORMAL_01", "Value_0h_uA": 10.10, "Value_24h_uA": 10.32, "PropagationDelay_0h_ns": 2.01, "PropagationDelay_24h_ns": 2.02
    }),
    ("Scenario 2: Strong Anomaly", {
        "Component_ID": "IC_ANOMALY_02", "Value_0h_uA": 10.05, "Value_24h_uA": 16.50, "PropagationDelay_0h_ns": 2.01, "PropagationDelay_24h_ns": 2.45
    }),
    ("Scenario 3: Latent Defect Below Datasheet Limit (Core SIH)", {
        "Component_ID": "IC_LATENT_ESCAPE_03", "Value_0h_uA": 10.12, "Value_24h_uA": 11.56, "PropagationDelay_0h_ns": 2.015, "PropagationDelay_24h_ns": 2.218
    }),
    ("Scenario 4: Hard Datasheet Limit Failure", {
        "Component_ID": "IC_HARD_FAIL_04", "Value_0h_uA": 38.00, "Value_24h_uA": 48.50, "PropagationDelay_0h_ns": 2.10, "PropagationDelay_24h_ns": 2.80
    }),
    ("Scenario 5: Missing Measurements", {
        "Component_ID": "IC_SPARSE_05", "Value_0h_uA": 10.15, "Value_24h_uA": 10.40
    }),
    ("Scenario 6: Extreme / Sensor Error", {
        "Component_ID": "IC_FAULT_06", "Value_0h_uA": -15.00, "Value_24h_uA": 1200.0
    }),
]

print("=" * 85)
print("END-TO-END INFERENCE PIPELINE DEMONSTRATION ACROSS 6 SCENARIOS:")
print("=" * 85)

for title, data in scenarios:
    res = analyze_component(data)
    cid = res["component_id"]
    lvl = res["risk_level"]
    score = res["risk_score"]
    ano = res["anomaly_score"]
    p168 = res["predicted_168h"]
    d_pass = res["datasheet_pass"]
    rate = res["predicted_drift_rate"]
    slp = bool(res["slope_exceeded"])
    reasons = res["explanation"]["human_readable_reasons"]
    
    print(f"\n>>> {title} ({cid})")
    print(f"    - Risk Level       : {lvl:10s} (Risk Score: {score:5.1f} / 100)")
    print(f"    - Anomaly Score    : {ano:5.4f}")
    print(f"    - Predicted 168h   : {p168:5.2f} uA (Datasheet Pass: {d_pass})")
    print(f"    - Drift Velocity   : {rate:5.4f} uA/h (Safety Slope Breached: {slp})")
    print(f"    - Primary Reason   : {reasons[0]}")
    if res["validation_warnings"]:
        print(f"    - Warnings Flagged : {res['validation_warnings']}")
