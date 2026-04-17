import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import json
from src.synthetic_data import generate_synthetic_ehr

# ToolResult class for consistency
class ToolResult:
    def __init__(self, content: list):
        self.content = content


def fhir_data_tool(patient_id: str):
    """MCP Tool: Retrieve synthetic FHIR-like patient record"""
    data = generate_synthetic_ehr(patient_id)
    return {
        "status": "success",
        "patient_id": patient_id,
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
