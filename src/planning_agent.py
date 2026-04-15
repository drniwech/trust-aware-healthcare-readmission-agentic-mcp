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
1. fhir_data_tool     → Retrieve synthetic (or real) FHIR patient EHR data.
2. predict_readmission_tool → Compute 30-day readmission risk probability.
3. explain_prediction_tool  → Generate SHAP explanations and trust calibration score.

## REQUIRED WORKFLOW:
Create a clear, numbered step-by-step plan that strictly follows this sequence:
1. Use fhir_data_tool to pull patient data via the research agent.
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
    """Main executor that runs the full multi-agent pipeline."""
    print(f"🚀 Starting trust-aware readmission prediction workflow using {model}...")

    # Step 1: Generate clinical workflow plan
    plan = planner_agent(prompt, model)

    # Step 2: Research phase using MCP healthcare tools
    research_output, _ = research_agent(
        f"Execute the following plan using MCP tools:\n{plan}\n\nOriginal request: {prompt}",
        model
    )

    # Step 3: Write clinical report
    draft, _ = writer_agent(research_output, model)

    # Step 4: Final clinical editing
    final_report = editor_agent(draft, prompt, model)

    print("✅ Workflow completed successfully.")
    return final_report
