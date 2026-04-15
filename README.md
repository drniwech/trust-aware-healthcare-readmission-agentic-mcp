# Trust-Aware Healthcare Readmission Prediction Platform (Agentic MCP)

Adapted from the DeepLearning.AI Agentic Workflow course repo.  
This is a **FastAPI + Postgres** single-container web app that uses **Agentic AI + Model Context Protocol (MCP)** tools to deliver trust-aware 30-day hospital readmission risk predictions.

The system autonomously:
- Pulls patient EHR data via synthetic FHIR MCP tool
- Runs readmission prediction
- Generates SHAP explanations + trust calibration scores
- Produces clinician-ready reports with uncertainty visualization support

A separate **Streamlit clinician dashboard** provides interactive SHAP waterfalls, risk gauges, and override feedback.

## Features

- `/` serves a simple UI to start a prediction task.
- `/generate_report` launches the threaded multi-agent workflow (planner → MCP research → writer → editor).
- `/task_progress/{task_id}` and `/task_status/{task_id}` for monitoring.
- **MCP integration**: Dynamic tool discovery for FHIR data, prediction, and explainability.
- **Trust-aware design**: SHAP + trust calibration built into every output.
- **Clinician Dashboard**: Streamlit UI with Plotly visualizations (run separately).

## Project Layout

```
.
├─ main.py                      # FastAPI main application
├─ streamlit_dashboard.py       # Clinician UI with uncertainty viz
├─ src/
│  ├─ planning_agent.py         # planner_agent(), executor_agent_step()
│  ├─ agents.py                 # research_agent (MCP), writer_agent, editor_agent
│  ├─ mcp_healthcare_tools.py   # FHIR, predict, explain tools (synthetic)
│  └─ synthetic_data.py         # MIMIC-style EHR generator
│  └─ research_tools.py         # tavily_search_tool, arxiv_search_tool, wikipedia_search_tool
├─ templates/
│  └─ index.html                # UI page rendered by "/"
├─ static/                      # optional static assets (css/js)
├─ docker/
│  └─ entrypoint.sh             # starts Postgres, prepares DB, then launches Uvicorn
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

> Make sure `templates/index.html` and (optionally) `static/` exist and are copied into the image.

---

## Prerequisites

* **Docker** (Desktop on Windows/macOS, or engine on Linux).
* **Setup Ollama (One-time)** 

* API keys stored in a `.env` file:

  ```
  # === Switch between providers here ===
  # Local & free (recommended for now)
  DEFAULT_MODEL=ollama:llama3.2
  # Uncomment for OpenAI
  # DEFAULT_MODEL=openai:gpt-4o-mini

  # OpenAI (only needed when using OpenAI)
  OPENAI_API_KEY=your-open-api-key

  # Ollama (usually no need to change)
  OLLAMA_BASE_URL=http://localhost:11434
  
  # Tavily
  TAVILY_API_KEY=your-tavily-api-key
  ```

* Python deps are installed by Docker from `requirements.txt`:

  * `fastapi`, `uvicorn`, `sqlalchemy`, `python-dotenv`, `jinja2`, `requests`, `wikipedia`, etc.
  * Plus any libs used by your `aisuite` client.

---

## Environment variables

The app **reads only `DATABASE_URL`** at startup.

* The container’s entrypoint sets a sane default for local dev:

  ```
  postgresql://app:local@127.0.0.1:5432/agentic_db
  ```
* To use Tavily:

  * Provide `TAVILY_API_KEY` (via `.env` or `-e`).

Optional (if you want to override defaults done by the entrypoint):

* `POSTGRES_USER` (default `app`)
* `POSTGRES_PASSWORD` (default `local`)
* `POSTGRES_DB` (default `agentic_db`)

---

## Local Virtual Environment Setup
# 1. Create the new virtual environment with Python 3.11
```bash
python3.11 -m venv agentic
```
# 2. Activate the environment
```bash
source agentic/bin/activate        # On macOS / Linux
```
# After activation, your prompt should change and show (agentic)

# 3. Upgrade pip (good practice)
```bash
pip install --upgrade pip
```

# 4. Install your project dependencies
```bash
# cd trust-aware-healthcare-readmission-agentic-mcp
# At project root directory (trust-aware-healthcare-readmission-agentic-mcp)  
pip install -r requirements.txt
```

# 5. Build & Run (FastAPI + Postgres)

```bash
docker build -t trust-aware-readmission-mcp .
docker run --rm -it -p 8000:8000 -p 5432:5432 --env-file .env trust-aware-readmission-mcp
```

You should see logs like:

```
🚀 Starting Postgres cluster 17/main...
✅ Postgres is ready
CREATE ROLE
CREATE DATABASE
🔗 DATABASE_URL=postgresql://app:local@127.0.0.1:5432/agentic_db
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

# 6. Open the app

* UI: [http://localhost:8000/](http://localhost:8000/)
* Try this prompt:
```
Predict trust-aware 30-day readmission risk for patient 12345. 
Use MCP tools to pull data, run prediction, generate SHAP explanation, 
and trust calibration score.
```

![Trust-Aware Healthcare Readmission Prediction Platform](https://github.com/drniwech/trust-aware-healthcare-readmission-agentic-mcp/blob/main/static/mainapp.png)  

# 7. Interactive API Document
A nice interactive API documentation you see at /docs is Swagger UI (specifically, Swagger UI powered by FastAPI).
FastAPI automatically generates a full OpenAPI specification from your route definitions (@app.get, @app.post, 
Pydantic models, etc.), and then renders it beautifully using Swagger UI.  

* Swagger UI (the nice one), Interactive API tester: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc, Alternative documentation style: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* Raw OpenAPI schema, Machine-readable spec: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## Clinician Dashboard (Streamlit)
In a separate terminal (after starting the API):  
  - Make sure streamlit is installed in your local environment:  
    ```bash
    pip install streamlit
    ```

Run Streamlit Dashboard: 
```bash
streamlit run streamlit_dashboard.py
```
You should see logs like:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.86.64:8501
```
Open: http://localhost:8501  

![Trust-Aware Healthcare Readmission Prediction Dashboard](https://github.com/drniwech/trust-aware-healthcare-readmission-agentic-mcp/blob/main/static/streamlit.png)

## Quick API Example

```bash
curl -X POST http://localhost:8000/generate_report \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate trust-aware readmission prediction for patient 12345", "model":"openai:gpt-4o-mini"}'
```

## Development & Customization

- Swap synthetic FHIR tool → real FHIR MCP server (e.g., Momentum or WSO2) by updating mcp_healthcare_tools.py.
- Add real ML models or LangGraph for more advanced orchestration. 
- Feedback from the dashboard improves future trust calibration (human-in-the-loop).  

## Research Use
This prototype demonstrates Agentic AI + MCP applied to healthcare readmission prediction with built-in trust and explainability. Cite as needed for papers on trust-aware clinical AI systems.
Built for the Agentic AI + MCP exploration.
