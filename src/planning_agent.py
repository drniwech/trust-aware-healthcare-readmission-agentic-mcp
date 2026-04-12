from datetime import datetime
from aisuite import Client

from src.agents import research_agent, writer_agent, editor_agent
from src.config import DEFAULT_MODEL

client = Client()

def planner_agent(prompt: str, model: str = DEFAULT_MODEL):
    print("==================================")
    print("Clinical Workflow Planner Agent")
    print("==================================")

    full_prompt = f"""
You are an expert clinical AI workflow planner for a **Trust-Aware Healthcare Readmission Prediction Platform**.

User request: {prompt}

## AVAILABLE MCP HEALTHCARE TOOLS (use via research_agent):
1. fhir_data_tool – Retrieve synthetic (or real) FHIR patient EHR data.
2. predict_readmission_tool – Compute 30-day readmission risk probability.
3. explain_prediction_tool – Generate SHAP explanations and trust calibration score.

## REQUIRED WORKFLOW PLAN:
Create a clear, numbered step-by-step plan that strictly follows this sequence for trust-aware prediction:
1. Pull patient data using fhir_data_tool via the research agent.
2. Run the prediction model using predict_readmission_tool.
3. Generate explainability and trust metrics using explain_prediction_tool.
4. Synthesize everything into a clinical report using the writer agent.
5. Perform final clinical review and editing.

Output ONLY a numbered list of steps with brief descriptions. Focus on trust calibration, explainability, and clinician actionability.

Today is {datetime.now().strftime("%Y-%m-%d")}.
"""

    messages = [{"role": "user", "content": full_prompt}]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        plan = resp.choices[0].message.content or ""
        print("Plan generated:\n", plan)
        return plan
    except Exception as e:
        print("Planner Error:", e)
        return "1. Pull FHIR data\n2. Run prediction\n3. Generate SHAP + trust explanation\n4. Write report\n5. Edit for clinical quality"


def execute_task(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Main executor that runs the full multi-agent pipeline (original pattern preserved)."""
    print("Starting trust-aware readmission prediction workflow...")

    # Step 1: Generate plan
    plan = planner_agent(prompt, model)

    # Step 2: Research phase (MCP healthcare tools)
    research_output, _ = research_agent(f"Execute the following plan using MCP tools:\n{plan}\n\nOriginal request: {prompt}", model)

    # Step 3: Write clinical report
    draft, _ = writer_agent(research_output, model)

    # Step 4: Edit for clinical quality and trust emphasis
    final_report = editor_agent(draft, prompt, model)

    print("Workflow completed successfully.")
    return final_report
