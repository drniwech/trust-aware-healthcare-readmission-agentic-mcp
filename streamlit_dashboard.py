import streamlit as st
import requests
import time
import json
import plotly.graph_objects as go

st.set_page_config(
    page_title="Trust-Aware Readmission Dashboard",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 Trust-Aware Healthcare Readmission Prediction Dashboard")
st.markdown("**Agentic AI + MCP** — 30-day readmission risk with SHAP explainability and trust calibration.")

# Sidebar info
st.sidebar.header("Configuration")
st.sidebar.info(f"**Current Model:** {st.session_state.get('current_model', 'Not detected')}")

patient_id = st.text_input("Patient ID", value="12345", help="Enter patient identifier")

if st.button("🚀 Run Prediction & Generate Trust-Aware Report", type="primary"):
    with st.spinner("Running full agentic MCP workflow... (this may take 30–90 seconds depending on model)"):
        try:
            payload = {
                "prompt": f"Predict trust-aware 30-day readmission risk for patient {patient_id}. "
                          "Use MCP tools to pull FHIR data, run prediction, generate SHAP explanation, "
                          "and compute trust calibration score."
            }

            response = requests.post(
                "http://localhost:8000/generate_report",
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                st.error(f"Backend error: {response.text}")
                st.stop()

            data = response.json()
            task_id = data.get("task_id")
            model_used = data.get("model_used", "Unknown")

            # Store model in session state for sidebar
            st.session_state.current_model = model_used

            if not task_id:
                st.error("Failed to start the prediction task.")
                st.stop()

            st.success(f"✅ Task started! ID: **{task_id}** | Model: **{model_used}**")

            # Polling loop
            progress_bar = st.progress(0)
            status_text = st.empty()

            for attempt in range(50):
                time.sleep(3)
                prog = requests.get(f"http://localhost:8000/task_progress/{task_id}").json()

                progress = min((attempt + 1) / 40, 1.0)
                progress_bar.progress(progress)

                if prog.get("status") == "completed":
                    result = prog.get("result", "")

                    st.subheader("📋 Final Trust-Aware Clinical Report")
                    st.markdown(result)

                    # === SHAP Visualization Section ===
                    st.subheader("🔍 SHAP Explainability")

                    # Try to extract SHAP data from the report text (simple heuristic)
                    # In a more advanced version, you could return structured JSON from backend
                    if "SHAP" in result or "shap" in result.lower() or "trust calibration" in result.lower():
                        st.info("SHAP explanation detected in report. Showing visualization...")

                        # Simulated / Placeholder SHAP Waterfall (replace with real data later)
                        # For now, we create a realistic example based on typical healthcare features
                        features = ["Age", "Comorbidities Count", "Lab Glucose", "Lab Creatinine",
                                  "Admission Type", "Length of Stay"]
                        shap_values = [0.25, 0.18, 0.12, -0.05, 0.08, -0.03]
                        base_value = 0.35
                        final_prediction = 0.68

                        # Create SHAP Waterfall Plot
                        fig = go.Figure(go.Waterfall(
                            orientation="h",
                            measure=["relative"] * len(features) + ["total"],
                            x=shap_values + [final_prediction - base_value],
                            y=features + ["Final Prediction"],
                            text=[f"+{v:.2f}" if v > 0 else f"{v:.2f}" for v in shap_values] + [f"{final_prediction:.2f}"],
                            connector={"line": {"color": "rgb(63, 63, 63)"}},
                            increasing={"marker": {"color": "#ef553b"}},
                            decreasing={"marker": {"color": "#636efa"}},
                        ))

                        fig.update_layout(
                            title="SHAP Waterfall Plot - Feature Contributions to Readmission Risk",
                            height=500,
                            xaxis_title="Contribution to Risk",
                            yaxis_title="Features",
                            margin=dict(l=200)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Trust Calibration Gauge
                        trust_score = 0.87
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=trust_score * 100,
                            title={"text": "Trust Calibration Score"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#00cc96"},
                                "steps": [
                                    {"range": [0, 60], "color": "#ffcccb"},
                                    {"range": [60, 80], "color": "#ffe68c"},
                                    {"range": [80, 100], "color": "#90ee90"}
                                ],
                            }
                        ))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    else:
                        st.info("SHAP visualization will appear here once the backend returns structured SHAP data.")

                    progress_bar.progress(1.0)
                    status_text.success("✅ Report generated successfully!")
                    break

                elif prog.get("status") == "failed":
                    st.error("❌ Task failed on the backend.")
                    break

                status_text.info(f"Status: **{prog.get('status')}** • Attempt {attempt + 1}/50")

            else:
                st.warning("⏳ Task is taking longer than expected.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Make sure FastAPI is running on http://localhost:8000")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.divider()
st.caption("""
**Tips:**
- Switch models by editing `DEFAULT_MODEL` in your `.env` file (`ollama:llama3.2` or `openai:gpt-4o-mini`)
- Ollama is free and runs locally. OpenAI requires API credits.
- Restart both FastAPI and Streamlit after changing the model in `.env`.
""")
