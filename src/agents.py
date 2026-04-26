import json
from datetime import datetime #It is used in the prompt templates
import os
from dotenv import load_dotenv

load_dotenv()

from src.config import DEFAULT_MODEL
# Using in the prompt templates, do not remove the following line.
from src.mcp_healthcare_tools import fhir_data_tool, predict_readmission_tool, explain_prediction_tool, patient_ehr_tool

# ====================== LLM SETUP ======================
if DEFAULT_MODEL.startswith("ollama:"):
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=DEFAULT_MODEL.replace("ollama:", ""),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.2,
        num_ctx=16384,
    )
    USE_OLLAMA = True
    print(f"✅ [Ollama] Using local model: {DEFAULT_MODEL}")
else:
    import aisuite as ai
    client = ai.Client()
    USE_OLLAMA = False
    print(f"✅ [OpenAI] Using cloud model: {DEFAULT_MODEL}")


# ====================== RESEARCH AGENT (MCP Healthcare Tools) ======================
def research_agent(prompt: str, model: str = DEFAULT_MODEL):
    print("==================================")
    print("Healthcare MCP Research Agent (Real HAPI FHIR)")
    print("==================================")

    full_prompt = f"""
You are a precise clinical AI assistant. Respond with **ONLY** valid JSON. No explanations, no markdown, no extra text.

User request: {prompt}

Use the tools in this exact order:
1. patient_ehr_tool(patient_id)
2. predict_readmission_tool(patient_data)
3. explain_prediction_tool(patient_data, prediction)

Return EXACTLY this JSON structure and nothing else:

{{
  "report_markdown": "Full clinician-friendly Markdown report here with sections for Patient Overview, Risk, SHAP, and Recommendations.",
  "structured_data": {{
    "patient_id": "12345",
    "readmission_risk": 0.68,
    "risk_percent": "68.0%",
    "trust_calibration_score": 0.87,
    "shap_values": {{"age": 0.25, "comorbidities_count": 0.18, "lab_glucose": 0.12, "lab_creatinine": -0.05}},
    "top_positive_factors": [["age", 0.25], ["comorbidities_count", 0.18]]
  }}
}}
"""

    try:
        if USE_OLLAMA:
            response = llm.invoke(full_prompt)
            content = response.content.strip()
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()

        # Robust JSON extraction - remove any text before first { and after last }
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            structured_output = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response")

        return structured_output

    except json.JSONDecodeError as je:
        print(f"JSON Parse Error: {je}")
        print(f"Raw LLM output was: {content[:500]}...")
        return {
            "report_markdown": f"[ERROR] Failed to parse JSON from model response. Raw output: {content[:300]}...",
            "structured_data": {}
        }
    except Exception as e:
        print(f"Research Agent Error: {e}")
        return {
            "report_markdown": f"[ERROR] {str(e)}",
            "structured_data": {}
        }


# ====================== WRITER AGENT ======================
def writer_agent(prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 12000):
    print("==================================")
    print("Clinical Report Writer Agent")
    print("==================================")

    system_message = """
You are an expert clinical informaticist. Write a professional, clear, and actionable report.
Emphasize trust calibration, SHAP explanations, and clinical recommendations.
Use Markdown formatting with clear sections.

Use ONLY the following structured patient data. Do NOT use placeholders like [Insert ...].

Patient Data:
{patient_data}

Readmission Risk Prediction: {risk_score}% (Low/Medium/High)
SHAP Key Factors: {shap_explanation}
Trust Calibration Score: {trust_score}

Write the report in this exact structure, filling in real values:

**Patient Information**
Name: [use real name]
Date of Birth: [use real DOB]
Primary Diagnosis: [use real diagnosis]

**Current Symptoms**
Chief Complaint: ...
Secondary Symptoms: ...

**Medical History**
Relevant Allergies: ...
Previous Surgical Procedures: ...
Medications: ...

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
        print(f"Writer Error: {e}")
        return f"[Writer Error] {str(e)}", []


# ====================== EDITOR AGENT ======================
def editor_agent(draft: str, prompt: str, model: str = DEFAULT_MODEL):
    print("==================================")
    print("Clinical Editor Agent")
    print("==================================")

    full_prompt = f"""
You are a senior clinician reviewing a trust-aware readmission prediction report.

Original request: {prompt}

Draft report:
{draft}

Improve clinical tone, clarity, and actionability. Strengthen sections on SHAP and trust calibration.
Return ONLY the final polished Markdown report.
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
        return draft  # fallback
