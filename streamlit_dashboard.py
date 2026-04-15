import streamlit as st
import requests
import time
import json

st.set_page_config(
    page_title="Trust-Aware Readmission Dashboard",
    layout="wide",
    page_icon="🩺"
)

st.title("🩺 Trust-Aware Healthcare Readmission Prediction Dashboard")
st.markdown("**Agentic AI + MCP** — Predicts 30-day readmission risk with SHAP explanations and trust calibration.")

# Show current model being used
st.sidebar.header("Configuration")
st.sidebar.info(f"**Current Model:** {st.session_state.get('current_model', 'Not detected')}")

patient_id = st.text_input("Patient ID", value="12345", help="Enter a patient identifier (e.g., 12345)")

if st.button("🚀 Run Prediction & Generate Trust-Aware Report", type="primary"):
    with st.spinner("Running full agentic MCP workflow... (this may take 30–90 seconds depending on model)"):
        try:
            payload = {
                "prompt": f"Predict trust-aware 30-day readmission risk for patient {patient_id}. "
                          "Use MCP tools to pull FHIR data, run prediction, generate SHAP explanation, "
                          "and compute trust calibration score. Provide a clear clinical report."
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

            # Store model in session state for sidebar
            st.session_state.current_model = model_used

            if not task_id:
                st.error("Failed to start the prediction task.")
                st.stop()

            st.success(f"✅ Task started! ID: **{task_id}** | Model: **{model_used}**")

            # Progress polling
            progress_bar = st.progress(0)
            status_text = st.empty()

            for attempt in range(50):  # Max ~2.5 minutes
                time.sleep(3)
                prog_response = requests.get(f"http://localhost:8000/task_progress/{task_id}")

                if prog_response.status_code != 200:
                    st.error("Failed to fetch task progress.")
                    break

                prog = prog_response.json()
                status = prog.get("status")

                progress = min((attempt + 1) / 40, 1.0)
                progress_bar.progress(progress)

                if status == "completed":
                    result = prog.get("result", "No result returned")
                    st.subheader("📋 Final Trust-Aware Clinical Report")
                    st.markdown(result)
                    progress_bar.progress(1.0)
                    status_text.success("✅ Report generated successfully!")
                    break

                elif status == "failed":
                    st.error("❌ Task failed on the backend. Check FastAPI logs for details.")
                    break

                status_text.info(f"Status: **{status}** • Attempt {attempt + 1}/50")

            else:
                st.warning("⏳ Task is taking longer than expected. Please check the FastAPI terminal logs.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the FastAPI backend.\n"
                     "Make sure the backend container is running on http://localhost:8000")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

# Footer / Help
st.divider()
st.caption("""
**Tips:**
- Switch models by editing `DEFAULT_MODEL` in your `.env` file (`ollama:llama3.2` or `openai:gpt-4o-mini`)
- Ollama is free and runs locally. OpenAI requires API credits.
- Restart both FastAPI and Streamlit after changing the model in `.env`.
""")
