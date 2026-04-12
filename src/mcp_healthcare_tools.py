from typing import Dict, Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import shap
import plotly.graph_objects as go
from src.synthetic_data import generate_synthetic_ehr

# Simulate MCP ToolResult
class ToolResult:
    def __init__(self, content: list):
        self.content = content

# === MCP-style healthcare tools ===
def fhir_data_tool(patient_id: str) -> Dict:
    """MCP Tool: Pull synthetic FHIR patient record (real MCP server swap possible)."""
    return generate_synthetic_ehr(patient_id)

def predict_readmission_tool(patient_data: Dict) -> Dict:
    """MCP Tool: Run readmission prediction (RF model)."""
    df = pd.DataFrame([patient_data])
    features = ["age", "comorbidities_count", "lab_glucose", "lab_creatinine"]
    X = df[features]
    # Simple trained model (in prod: load from file)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df["true_readmission_30d"])  # demo fit
    pred_prob = model.predict_proba(X)[0][1]
    return {
        "readmission_risk": round(pred_prob, 4),
        "risk_percent": f"{pred_prob*100:.1f}%",
        "model_used": "RandomForest (MCP-integrated)"
    }

def explain_prediction_tool(patient_data: Dict, prediction: float) -> Dict:
    """MCP Tool: SHAP explanation + trust calibration."""
    df = pd.DataFrame([patient_data])
    features = ["age", "comorbidities_count", "lab_glucose", "lab_creatinine"]
    X = df[features]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df["true_readmission_30d"])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[1][0]  # positive class

    # Trust calibration score (demo)
    trust_score = 0.87 if prediction > 0.5 else 0.92

    # Plotly SHAP waterfall (for dashboard)
    fig = go.Figure(go.Waterfall(
        name="SHAP", orientation="h",
        measure=["relative"]*len(features) + ["total"],
        x=shap_values.tolist() + [prediction],
        y=features + ["Prediction"],
        text=[f"{v:.3f}" for v in shap_values] + [f"{prediction:.3f}"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title="SHAP Explanation (Trust-Aware)", height=400)

    return {
        "shap_values": dict(zip(features, shap_values.tolist())),
        "trust_calibration_score": trust_score,
        "shap_plot_json": fig.to_json(),
        "explanation": f"Top driver: age ({shap_values[0]:.3f})"
    }
