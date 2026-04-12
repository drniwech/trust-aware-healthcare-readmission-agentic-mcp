from datetime import datetime
from aisuite import Client
from src.mcp_healthcare_tools import (
    fhir_data_tool,
    predict_readmission_tool,
    explain_prediction_tool,
)

client = Client()

# === Research Agent (now Healthcare Data + MCP Tool Agent) ===
def research_agent(
    prompt: str, model: str = "openai:gpt-4o-mini", return_messages: bool = False
):
    print("==================================")
    print("Healthcare MCP Research Agent")
    print("==================================")

    full_prompt = f"""
You are an advanced clinical research assistant specialized in hospital readmission prediction and trust-aware AI systems. 
Your mission is to gather comprehensive patient data and run predictive modeling using MCP tools for a trust-aware 30-day readmission risk assessment.

## AVAILABLE MCP HEALTHCARE TOOLS:

1. **`fhir_data_tool`**: Retrieves synthetic (or real FHIR) patient EHR record.
   - USE FOR: Pulling patient demographics, labs, comorbidities, notes, etc.

2. **`predict_readmission_tool`**: Runs the readmission prediction model on patient data.
   - USE FOR: Generating 30-day readmission risk probability.

3. **`explain_prediction_tool`**: Generates SHAP explanations and trust calibration score.
   - USE FOR: Explainability, uncertainty, and trust metrics for clinicians.

## WORKFLOW METHODOLOGY:

1. **Analyze Request**: Identify patient ID or cohort and key clinical questions.
2. **Pull Data**: Always start with fhir_data_tool to get EHR context.
3. **Run Prediction**: Call predict_readmission_tool with the retrieved data.
4. **Generate Explanations**: Use explain_prediction_tool for SHAP + trust score.
5. **Synthesize Findings**: Combine results into a clinician-ready summary with risk, explanations, and trust calibration.

## TOOL SELECTION GUIDELINES:
- Always use fhir_data_tool first for any patient-specific request.
- Follow with predict_readmission_tool.
- Always end with explain_prediction_tool to ensure trust-awareness.
- For cohort requests, apply tools iteratively or in batch.

## OUTPUT FORMAT:
Present findings in a structured clinical format:
1. **Patient Summary**: Key demographics and data pulled via MCP.
2. **Readmission Risk**: Probability and percentage.
3. **Trust-Aware Explanation**: SHAP drivers and trust calibration score.
4. **Clinical Recommendations**: Actionable insights and limitations.
5. **Tools Used**: List of MCP tools invoked.

Today is {datetime.now().strftime("%Y-%m-%d")}.

USER REQUEST:
{prompt}
""".strip()

    messages = [{"role": "user", "content": full_prompt}]
    tools = [fhir_data_tool, predict_readmission_tool, explain_prediction_tool]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_turns=5,
            temperature=0.0,  # deterministic for clinical reproducibility
        )

        content = resp.choices[0].message.content or ""

        # Collect tool calls (same logic as original repo)
        calls = []

        for ir in getattr(resp, "intermediate_responses", []) or []:
            try:
                tcs = ir.choices[0].message.tool_calls or []
                for tc in tcs:
                    calls.append((tc.function.name, tc.function.arguments))
            except Exception:
                pass

        for msg in getattr(resp.choices[0].message, "intermediate_messages", []) or []:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append((tc.function.name, tc.function.arguments))

        # Deduplicate while preserving order
        seen = set()
        dedup_calls = []
        for name, args in calls:
            key = (name, args)
            if key not in seen:
                seen.add(key)
                dedup_calls.append((name, args))

        # Pretty-print tool usage
        tool_lines = []
        for name, args in dedup_calls:
            arg_text = str(args)
            tool_lines.append(f"- {name}({arg_text})")

        if tool_lines:
            tools_html = "\n\n## MCP Tools Used\n"
            tools_html += "\n".join([f"* {line}" for line in tool_lines])
            content += tools_html

        print("Output:\n", content)
        return content, messages

    except Exception as e:
        print("Error:", e)
        return f"[MCP Agent Error: {str(e)}]", messages


# === Writer Agent (adapted for clinical trust-aware report) ===
def writer_agent(
    prompt: str,
    model: str = "openai:gpt-4o-mini",
    min_words_total: int = 1800,
    min_words_per_section: int = 300,
    max_tokens: int = 12000,
    retries: int = 1,
):
    print("==================================")
    print("Clinical Report Writer Agent")
    print("==================================")

    system_message = """
You are an expert clinical informaticist and academic writer specializing in AI-driven healthcare decision support systems. 
Your task is to synthesize MCP tool outputs (FHIR data, prediction, SHAP explanations, trust scores) into a polished, clinician-ready report on trust-aware readmission prediction.

## REPORT REQUIREMENTS:
- Produce a COMPLETE, PROFESSIONAL, and CLINICALLY ACTIONABLE report in Markdown format.
- Emphasize trust calibration, explainability, and clinician override recommendations.
- Length should thoroughly cover the patient/cohort case (typically 1200-2500 words).

## MANDATORY STRUCTURE:
1. **Title**: e.g., "Trust-Aware 30-Day Readmission Risk Assessment for Patient [ID]"
2. **Abstract**: Summary of risk, trust score, and key drivers.
3. **Patient Overview**: Demographics and key EHR findings from FHIR MCP.
4. **Prediction Results**: Readmission probability with confidence.
5. **Trust-Aware Explainability**: SHAP analysis and trust calibration score.
6. **Clinical Interpretation & Recommendations**: Actionable insights and limitations.
7. **Uncertainty & Override Guidance**: When and how clinicians should intervene.
8. **References & Tools**: MCP tools used and data sources.

## WRITING GUIDELINES:
- Use formal, precise, objective clinical language.
- Support every claim with MCP-derived evidence and citations.
- Highlight trust calibration (e.g., "Trust score: 0.89 – high reliability").
- Include recommendations for human-in-the-loop override to improve future model trust.
- Use Markdown for formatting; include placeholders for SHAP visualizations if data available.

Output ONLY the complete report in Markdown. No meta-commentary.
"""

    full_prompt = f"{system_message}\n\nRESEARCH MATERIALS TO SYNTHESIZE:\n{prompt}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": full_prompt},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""
        print("Writer output length:", len(content))
        return content, messages

    except Exception as e:
        print("Writer Error:", e)
        return f"[Writer Error: {str(e)}]", messages


# === Editor Agent (adapted for clinical quality & trust) ===
def editor_agent(
    draft: str,
    prompt: str,
    model: str = "openai:gpt-4o-mini",
):
    print("==================================")
    print("Clinical Editor Agent")
    print("==================================")

    full_prompt = f"""
You are a senior clinician and medical editor reviewing a trust-aware readmission prediction report.

ORIGINAL USER REQUEST:
{prompt}

DRAFT REPORT:
{draft}

## EDITING TASK:
Improve the draft for clinical accuracy, clarity, trust emphasis, and actionability. 
- Strengthen sections on SHAP explanations and trust calibration.
- Ensure recommendations support safe human-AI collaboration.
- Fix any factual inconsistencies with typical MCP outputs.
- Maintain professional tone suitable for hospital use.

Return the fully edited, final version of the report in Markdown format only.
"""

    messages = [{"role": "user", "content": full_prompt}]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
        print("Editor output length:", len(content))
        return content

    except Exception as e:
        print("Editor Error:", e)
        return draft  # fallback to original draft