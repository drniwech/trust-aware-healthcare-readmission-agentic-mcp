import os
import threading
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.planning_agent import execute_task
from src.config import DEFAULT_MODEL   # Import the centralized model config

load_dotenv()

app = FastAPI(title="Trust-Aware Healthcare Readmission Prediction Platform")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:local@localhost:5432/agentic_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text)
    status = Column(String, default="pending")
    result = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

class ReportRequest(BaseModel):
    prompt: str
    model: str = DEFAULT_MODEL      # Uses centralized config from .env / config.py


def run_task_in_thread(task_id: int, prompt: str, model: str):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "running"
        db.commit()

        # Execute the full agentic workflow
        result = execute_task(prompt, model)

        task.result = result
        task.status = "completed"
        task.updated_at = datetime.utcnow()
        db.commit()
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        task.status = "failed"
        task.result = error_details
        db.commit()
        print("=== THREAD ERROR ===")
        print(error_details)
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page using the updated TemplateResponse signature."""
    # Clear cache to avoid any lingering Jinja2 issues
    if hasattr(templates.env, "cache") and templates.env.cache is not None:
        templates.env.cache.clear()

    # Modern correct call: pass request first, then template name
    return templates.TemplateResponse(
        request=request,  # ← Required first positional argument in new Starlette
        name="index.html",
        context={}  # context is optional; request is already passed
    )


@app.post("/generate_report")
async def generate_report(request: ReportRequest):
    db = SessionLocal()
    try:
        task = Task(prompt=request.prompt, status="pending")
        db.add(task)
        db.commit()
        db.refresh(task)

        thread = threading.Thread(
            target=run_task_in_thread,
            args=(task.id, request.prompt, request.model),
            daemon=True,
        )
        thread.start()

        return {"task_id": task.id, "status": "pending", "model_used": request.model}
    finally:
        db.close()


@app.get("/task_progress/{task_id}")
async def task_progress(task_id: int):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return {"status": "not_found"}
        return {
            "task_id": task.id,
            "status": task.status,
            "result": task.result if task.status == "completed" else None,
            "created_at": task.created_at,
            "model_used": DEFAULT_MODEL,
        }
    finally:
        db.close()


@app.get("/task_status/{task_id}")
async def task_status(task_id: int):
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return {"status": "not_found"}
        return {"status": task.status}
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
