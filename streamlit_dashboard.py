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
st.markdown("**Agentic AI + MCP** — 30-day readmission risk with **dynamic SHAP explainability**")

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.info(f"**Current Model:** {st.session_state.get('current_model', 'Not detected')}")

patient_id = st.text_input("Patient ID", value="12345", help="Enter a patient identifier")

if st.button("🚀 Run Prediction & Generate Trust-Aware Report", type="primary"):
    with st.spinner("Running full agentic MCP workflow... (this may take 30–90 seconds depending on model)"):
        try:
            payload = {
                "prompt": f"Predict trust-aware 30-day readmission risk for patient {patient_id}. "
                          "Use MCP tools and return structured SHAP data."
            }

            response = requests.post(
                "http://localhost:8000/generate_report",
                json=payload,
                timeout=180
            )

            if response.status_code != 200:
                st.error(f"Backend returned error: {response.text}")
                st.stop()

            data = response.json()
            task_id = data.get("task_id")
            model_used = data.get("model_used", "Unknown")

            st.session_state.current_model = model_used

            if not task_id:
                st.error("Failed to start task.")
                st.stop()

            st.success(f"✅ Task started! ID: **{task_id}** | Model: **{model_used}**")

            # Polling for result
            progress_bar = st.progress(0)
            status_text = st.empty()

            for attempt in range(50):  # Max ~2.5 minutes
                time.sleep(3)
                prog_response = requests.get(f"http://localhost:8000/task_progress/{task_id}")
                prog = prog_response.json()

                progress = min((attempt + 1) / 40, 1.0)
                progress_bar.progress(progress)

                if prog.get("status") == "completed":
                    result = prog.get("result", "")

                    # Parse structured JSON output from backend
                    try:
                        result_dict = json.loads(result)
                        report_markdown = result_dict.get("report_markdown", result)
                        structured = result_dict.get("structured_data", {})

                        st.subheader("📋 Final Trust-Aware Clinical Report")
                        st.markdown(report_markdown)

                        # === Dynamic SHAP Visualization ===
                        if structured and "shap_values" in structured:
                            st.subheader("🔍 Dynamic SHAP Explainability")

                            shap_dict = structured["shap_values"]
                            risk_percent = structured.get("readmission_risk", 0.5)
                            trust_score = structured.get("trust_calibration_score", 0.85)

                            features = list(shap_dict.keys())
                            shap_values = list(shap_dict.values())

                            # SHAP Waterfall Plot
                            fig = go.Figure(go.Waterfall(
                                orientation="h",
                                measure=["relative"] * len(features) + ["total"],
                                x=shap_values + [risk_percent - 0.35],
                                y=features + ["Final Readmission Risk"],
                                text=[f"{v:+.3f}" for v in shap_values] + [f"{risk_percent:.2f}"],
                                connector={"line": {"color": "rgb(63, 63, 63)"}},
                                increasing={"marker": {"color": "#ef553b"}},
                                decreasing={"marker": {"color": "#636efa"}},
                            ))

                            fig.update_layout(
                                title="SHAP Waterfall Plot - Feature Contributions to Readmission Risk",
                                height=520,
                                xaxis_title="Contribution to Predicted Risk",
                                yaxis_title="Clinical Features",
                                margin=dict(l=200, r=50)
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            # Trust Calibration Gauge
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
                            st.info("SHAP data not available in this run. The backend is still generating structured output.")

                    except (json.JSONDecodeError, TypeError):
                        # Fallback if output is not JSON
                        st.subheader("📋 Final Trust-Aware Clinical Report")
                        st.markdown(result)
                        st.info("SHAP visualization will appear once the backend returns structured data.")

                    progress_bar.progress(1.0)
                    status_text.success("✅ Report generated successfully!")
                    break

                elif prog.get("status") == "failed":
                    st.error("❌ Task failed on the backend. Check FastAPI terminal logs for details.")
                    break

                status_text.info(f"Status: **{prog.get('status')}** • Attempt {attempt + 1}/50")

            else:
                st.warning("⏳ Task is taking longer than expected. Check the FastAPI logs.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the FastAPI backend.\nMake sure the backend is running on http://localhost:8000")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.divider()
st.caption("""
**Tips:**
- Switch models by editing `DEFAULT_MODEL` in your `.env` file (`ollama:llama3.2` or `openai:gpt-4o-mini`)
- Ollama is free and runs locally. OpenAI requires API credits.
- Restart both FastAPI and Streamlit after changing the model in `.env`.
""")
