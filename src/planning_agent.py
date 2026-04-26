import json
import os
from datetime import datetime

from src.agents import research_agent, writer_agent, editor_agent
from src.config import DEFAULT_MODEL


def planner_agent(prompt: str, model: str = DEFAULT_MODEL):
    print("==================================")
    print("Clinical Workflow Planner Agent")
    print("==================================")

    full_prompt = f"""
You are an expert clinical AI workflow planner for a Trust-Aware Healthcare Readmission Prediction Platform.

User request: {prompt}

## AVAILABLE MCP HEALTHCARE TOOLS:
1. patient_ehr_tool     → Retrieve real FHIR patient EHR data.
2. predict_readmission_tool → Compute 30-day readmission risk probability.
3. explain_prediction_tool  → Generate SHAP explanations and trust calibration score.

## REQUIRED WORKFLOW:
Create a clear, numbered step-by-step plan that strictly follows this sequence:
1. Use patient_ehr_tool to pull patient data via the research agent.
2. Run predict_readmission_tool to compute the risk.
3. Use explain_prediction_tool to generate SHAP explanation and trust calibration.
4. Hand off to the writer agent to synthesize a clinical report.
5. Perform final clinical review and editing.

Focus on trust calibration, explainability, and clinician actionability.

Today is {datetime.now().strftime("%Y-%m-%d")}.
Output ONLY a numbered list of steps.
"""

    # Use the same LLM logic as agents.py for consistency
    if DEFAULT_MODEL.startswith("ollama:"):
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=DEFAULT_MODEL.replace("ollama:", ""),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.1,
        )
        response = llm.invoke(full_prompt)
        plan = response.content
    else:
        import aisuite as ai
        client = ai.Client()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.1,
        )
        plan = resp.choices[0].message.content or ""

    print("Plan generated:\n", plan)
    return plan


def execute_task(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Main executor that runs the full multi-agent pipeline and returns structured JSON."""
    print(f"🚀 Starting trust-aware readmission prediction workflow using {model}...")

    # Step 1: Generate clinical workflow plan
    plan = planner_agent(prompt, model)

    # Step 2: Research phase (returns structured output)
    research_output = research_agent(
        f"Execute the following plan using MCP tools:\n{plan}\n\nOriginal request: {prompt}",
        model
    )
    # REMOVE
    print("Research output:\n", research_output)

    # Step 3: Write clinical report
    draft, _ = writer_agent(research_output.get("report_markdown", str(research_output)), model)
    #REMOVE
    print("Draft output:\n", draft)
    # Step 4: Final clinical editing
    final_report = editor_agent(draft, prompt, model)
    #REMOVE
    print("Final report:\n", final_report)
    # Combine everything into structured output
    final_output = {
        "report_markdown": final_report,
        "structured_data": research_output.get("structured_data", {})
    }
    #REMOVE
    print("Writer output:\n", final_output)
    return final_output
    # Return as JSON string so Streamlit can easily parse it
    return json.dumps(final_output)
