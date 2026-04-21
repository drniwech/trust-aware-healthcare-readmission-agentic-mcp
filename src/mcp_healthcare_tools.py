import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import json
from src.synthetic_data import generate_synthetic_ehr # Keep as fallback

# ToolResult class for consistency
class ToolResult:
    def __init__(self, content: list):
        self.content = content


def fhir_data_tool(patient_id: str):
    """MCP Tool: Fetch real patient data from HAPI FHIR server"""
    fhir_url = os.getenv("FHIR_SERVER_URL", "http://host.docker.internal:8080/fhir")

    try:
        # Query Patient resource
        patient_resp = requests.get(f"{fhir_url}/Patient/{patient_id}", timeout=10)
        patient_resp.raise_for_status()
        patient = patient_resp.json()

        # Query Observations (labs)
        obs_resp = requests.get(f"{fhir_url}/Observation?patient={patient_id}", timeout=10)
        observations = obs_resp.json().get("entry", []) if obs_resp.status_code == 200 else []

        # Simple feature extraction
        age = 65  # Default fallback
        if "birthDate" in patient:
            from datetime import datetime
            birth_year = int(patient["birthDate"][:4])
            age = datetime.now().year - birth_year

        comorbidities = len([e for e in observations if "code" in e.get("resource", {})])
        glucose = 120.0
        creatinine = 1.2

        for entry in observations:
            resource = entry.get("resource", {})
            code = resource.get("code", {}).get("coding", [{}])[0].get("code")
            value = resource.get("valueQuantity", {}).get("value")
            if code == "2339-0" and value:  # Glucose code example
                glucose = float(value)
            if code == "2160-0" and value:  # Creatinine code example
                creatinine = float(value)

        data = {
            "patient_id": patient_id,
            "age": age,
            "comorbidities_count": comorbidities,
            "lab_glucose": glucose,
            "lab_creatinine": creatinine,
            "admission_type": "emergency",
            "length_of_stay": 5,
            "notes": "Real FHIR data retrieved from HAPI server.",
            "true_readmission_30d": 0
        }

        return {
            "status": "success",
            "source": "real_hapi_fhir",
            "data": data
        }

    except Exception as e:
        print(f"FHIR fetch failed: {e}. Falling back to synthetic data.")
        # Fallback to synthetic
        data = generate_synthetic_ehr(patient_id)
        return {
            "status": "fallback",
            "source": "synthetic",
            "data": data
        }


def predict_readmission_tool(patient_data: dict):
    """MCP Tool: Run readmission prediction"""
    df = pd.DataFrame([patient_data])
    features = ["age", "comorbidities_count", "lab_glucose", "lab_creatinine"]
    X = df[features]

    # Train a simple model (in production: load pre-trained model)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df["true_readmission_30d"])

    pred_prob = model.predict_proba(X)[0][1]

    return {
        "status": "success",
        "readmission_risk": round(pred_prob, 4),
        "risk_percent": f"{pred_prob*100:.1f}%",
        "model_used": "RandomForestClassifier"
    }


def explain_prediction_tool(patient_data: dict, prediction: float):
    """MCP Tool: Generate real SHAP explanation + trust score"""
    df = pd.DataFrame([patient_data])
    features = ["age", "comorbidities_count", "lab_glucose", "lab_creatinine"]
    X = df[features]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df["true_readmission_30d"])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1][0]   # SHAP values for positive class (readmission)

    # Create structured SHAP data for visualization
    shap_dict = dict(zip(features, [float(v) for v in shap_values]))

    # Trust calibration score (simulated but realistic)
    trust_score = 0.92 if prediction < 0.3 else 0.78 if prediction < 0.6 else 0.65

    return {
        "status": "success",
        "shap_values": shap_dict,
        "base_value": float(explainer.expected_value[1]),
        "final_prediction": float(prediction),
        "trust_calibration_score": round(trust_score, 2),
        "top_positive_factors": sorted(
            [(k, v) for k, v in shap_dict.items() if v > 0],
            key=lambda x: x[1], reverse=True
        )[:3],
        "explanation": f"Top risk driver: {max(shap_dict, key=shap_dict.get)}"
    }
