from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

from src.config import DEFAULT_MODEL

# ====================== LLM SETUP ======================
if DEFAULT_MODEL.startswith("ollama:"):
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=DEFAULT_MODEL.replace("ollama:", ""),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.2,
        num_ctx=16384,          # Large context for agentic workflows
        num_predict=-1,         # No output token limit
    )
    print(f"✅ [Ollama] Using local model: {DEFAULT_MODEL}")
    USE_OLLAMA = True
else:
    # Use aisuite for OpenAI (and future cloud providers)
    import aisuite as ai
    client = ai.Client()
    print(f"✅ [OpenAI] Using cloud model: {DEFAULT_MODEL}")
    USE_OLLAMA = False


# ====================== RESEARCH AGENT (MCP Healthcare Tools) ======================
def research_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    return_messages: bool = False
):
    print("==================================")
    print("Healthcare MCP Research Agent")
    print("==================================")

    full_prompt = f"""
You are an advanced clinical research assistant specialized in trust-aware hospital readmission prediction.

## AVAILABLE MCP HEALTHCARE TOOLS:
1. fhir_data_tool → Retrieve synthetic FHIR patient EHR record.
2. predict_readmission_tool → Run 30-day readmission prediction model.
3. explain_prediction_tool → Generate SHAP explanations and trust calibration score.

## TASK:
{ prompt }

Follow this sequence:
1. Call fhir_data_tool to get patient data.
2. Call predict_readmission_tool.
3. Call explain_prediction_tool for explainability and trust metrics.
4. Synthesize all results into a clear clinical summary.

Today is {datetime.now().strftime("%Y-%m-%d")}.
"""

    try:
        if USE_OLLAMA:
            response = llm.invoke(full_prompt)
            content = response.content
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content or ""

        # Simple tool call logging (can be enhanced later)
        print("Research Agent completed.")

        return content, []

    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "429" in error_msg:
            friendly_error = "OpenAI quota exceeded. Please add credit at https://platform.openai.com/settings/organization/billing/overview"
        elif "invalid_api_key" in error_msg.lower() or "401" in error_msg:
            friendly_error = "Invalid OpenAI API key. Please check your .env file."
        else:
            friendly_error = f"Research Agent Error: {error_msg}"

        print(friendly_error)
        return f"[ERROR] {friendly_error}", []


# ====================== WRITER AGENT ======================
def writer_agent(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 12000,
):
    print("==================================")
    print("Clinical Report Writer Agent")
    print("==================================")

    system_message = """
You are an expert clinical informaticist and academic writer specializing in AI-driven healthcare decision support systems.

Write a professional, clinician-ready report on trust-aware 30-day hospital readmission prediction.
Emphasize SHAP explanations, trust calibration scores, and human-in-the-loop recommendations.

MANDATORY STRUCTURE:
1. Title
2. Abstract
3. Patient Overview
4. Prediction Results
5. Trust-Aware Explainability (SHAP + Trust Score)
6. Clinical Interpretation & Recommendations
7. Uncertainty & Override Guidance
8. MCP Tools Used

Use formal clinical language. Support every claim with evidence from the research materials.
"""

    full_prompt = f"{system_message}\n\nRESEARCH MATERIALS TO SYNTHESIZE:\n{prompt}"

    try:
        if USE_OLLAMA:
            response = llm.invoke(full_prompt)
            content = response.content
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""

        print(f"Writer output length: {len(content)} characters")
        return content, []

    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "429" in error_msg:
            friendly_error = "OpenAI quota exceeded. Please add credit."
        else:
            friendly_error = f"Writer Error: {error_msg}"

        print(friendly_error)
        return f"[ERROR] {friendly_error}", []


# ====================== EDITOR AGENT ======================
def editor_agent(
    draft: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
):
    print("==================================")
    print("Clinical Editor Agent")
    print("==================================")

    full_prompt = f"""
You are a senior clinician and medical editor reviewing a trust-aware readmission prediction report.

Original request: {prompt}

Draft report:
{draft}

Improve the draft for clinical accuracy, clarity, trust emphasis, and actionability.
Strengthen sections on SHAP explanations and trust calibration.
Ensure recommendations support safe human-AI collaboration.
Return ONLY the fully edited final report in clean Markdown format.
"""

    try:
        if USE_OLLAMA:
            response = llm.invoke(full_prompt)
            content = response.content
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.1,
            )
            content = resp.choices[0].message.content or ""

        print(f"Editor output length: {len(content)} characters")
        return content

    except Exception as e:
        print(f"Editor Error: {e}")
        return draft  # fallback to original draft
