import streamlit as st
import requests
import time

st.set_page_config(page_title="Trust-Aware Readmission Dashboard", layout="wide")
st.title("🩺 Trust-Aware Healthcare Readmission Prediction Dashboard")
st.markdown("**Agentic MCP Backend** — Predict 30-day readmission risk with SHAP explanations and trust calibration.")

patient_id = st.text_input("Patient ID", value="12345")

if st.button("🚀 Run Prediction & Get Trust-Aware Report"):
    with st.spinner("Running full agentic MCP workflow... (this may take 20–50 seconds)"):
        try:
            # Call the FastAPI backend
            response = requests.post(
                "http://localhost:8000/generate_report",
                json={
                    "prompt": f"Predict trust-aware 30-day readmission risk for patient {patient_id}. "
                              "Use MCP tools to pull FHIR data, run prediction, generate SHAP explanation, "
                              "and compute trust calibration score."
                },
                timeout=120
            )

            if response.status_code != 200:
                st.error(f"Backend error: {response.text}")
                st.stop()

            data = response.json()
            task_id = data.get("task_id")

            if not task_id:
                st.error("Failed to start task.")
                st.stop()

            st.success(f"Task started! ID: {task_id}. Waiting for result...")

            # Poll for completion
            for attempt in range(40):  # Max ~2 minutes
                time.sleep(3)
                prog_response = requests.get(f"http://localhost:8000/task_progress/{task_id}")
                prog = prog_response.json()

                if prog.get("status") == "completed":
                    result = prog.get("result", "No result returned")
                    st.subheader("📋 Final Trust-Aware Clinical Report")
                    st.markdown(result)
                    break

                elif prog.get("status") == "failed":
                    st.error("Task failed on the backend.")
                    break
            else:
                st.warning("Task is taking longer than expected. Check the FastAPI terminal logs.")

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Make sure the FastAPI app is running on http://localhost:8000")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.caption("💡 Tip: Add OpenAI credit to see full predictions. The report will include SHAP explanation and trust score when successful.")
