import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Trust-Aware Readmission Platform", layout="wide")
st.title("🩺 Trust-Aware Healthcare Readmission Prediction")

patient_id = st.text_input("Patient ID", "12345")
if st.button("Run Prediction"):
    resp = requests.post("http://localhost:8000/generate_report", json={"prompt": f"Predict readmission for patient {patient_id}"})
    task_id = resp.json()["task_id"]
    st.info(f"Task started: {task_id}")

    # Poll progress (same as FastAPI UI)
    # ... (full polling code)

    # Final dashboard
    result = {"readmission_risk": 0.68, "trust_score": 0.89, "shap_plot_json": "..."}
    st.metric("30-day Readmission Risk", f"{result['readmission_risk']*100:.1f}%")
    st.metric("Trust Calibration Score", f"{result['trust_score']*100:.0f}%")

    # SHAP visualization
    fig = go.Figure(...)  # loaded from shap_plot_json
    st.plotly_chart(fig, use_container_width=True)

    st.success("Clinician override option available – feedback improves future trust calibration.")
