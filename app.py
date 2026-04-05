"""
app.py
------
FastAPI backend for the Apollo MIS RAG system.

Endpoints:
  GET  /           → serves the frontend HTML
  GET  /health     → health check
  POST /query      → runs the full pipeline, returns JSON
  GET  /suggest    → returns example questions

Run with:
  uvicorn app:app --reload --port 8000
"""

import base64
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import our pipeline modules 
sys.path.insert(0, str(Path(__file__).parent))
from database import build_database, run_query
from kpi_retrieval import KPIRetriever
from sql_generator import SQLGenerator, SQLValidationError
from visualizer import visualize

# Startup 
print("[APP] Initialising Apollo RAG system...")
load_dotenv()
conn      = build_database()
retriever = KPIRetriever()

api_key = os.environ.get("GROQ_API_KEY")
generator = SQLGenerator(api_key=api_key) if api_key else None
if not generator:
    print("[APP] WARNING: ANTHROPIC_API_KEY not set — SQL generation disabled.")

app = FastAPI(title="Apollo MIS RAG System", version="1.0.0")

# Serve the charts folder as static files
charts_dir = Path(__file__).parent / "charts"
charts_dir.mkdir(exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(charts_dir)), name="charts")

# Models
class QueryRequest(BaseModel):
    question: str

# ENDPOINTS

@app.get("/health")
def health():
    return {
        "status": "ok",
        "sql_generation": "enabled" if generator else "disabled (no API key)",
        "kpi_index_size": len(retriever.kpis),
    }


EXAMPLE_QUESTIONS = [
    "Which athletes had the highest workload overall?",
    "Show average sprint distance by position",
    "Who has the most high intensity efforts?",
    "What is the average fatigue score per athlete?",
    "Show total distance for each athlete",
    "Which session type has the highest average distance?",
    "Who has the highest sprint distance on average?",
]

@app.get("/suggest")
def suggest():
    return {"questions": EXAMPLE_QUESTIONS}


@app.post("/query")
async def query_pipeline(req: QueryRequest):
    """
    Full RAG pipeline:
      1. KPI retrieval
      2. SQL generation (via Claude)
      3. SQL execution
      4. Visualization
    Returns structured JSON with all pipeline steps.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = {
        "question": question,
        "kpi_matches": [],
        "sql": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "chart_url": None,
        "error": None,
        "steps": [],
    }

    try:
        # Step 1: KPI Retrieval
        matches = retriever.retrieve(question, top_k=2)
        result["kpi_matches"] = [
            {"name": m.kpi_name, "description": m.description, "score": round(m.score, 3)}
            for m in matches
        ]
        kpi_context = retriever.format_for_prompt(matches)
        result["steps"].append("kpi_retrieval")

        # Step 2: SQL Generation
        if not generator:
            result["error"] = "SQL generation disabled — set ANTHROPIC_API_KEY"
            return result

        sql, df = generator.generate_with_retry(question, kpi_context, conn)

        if not sql:
            result["error"] = "Could not generate valid SQL for this question."
            return result

        result["sql"] = sql
        result["steps"].append("sql_generation")

        # Step 3: Results
        if df is None or df.empty:
            result["error"] = "Query returned no results. Try broadening your question."
            result["steps"].append("sql_execution")
            return result

        result["columns"]  = list(df.columns)
        result["rows"]     = df.values.tolist()
        result["row_count"] = len(df)
        result["steps"].append("sql_execution")

        # Step 4: Visualization
        chart_path = visualize(df, question)
        if chart_path and chart_path.exists():
            result["chart_url"] = f"/charts/{chart_path.name}"
        result["steps"].append("visualization")

    except SQLValidationError as e:
        result["error"] = f"SQL validation failed: {e}"
    except Exception as e:
        result["error"] = f"Pipeline error: {e}"
        print(traceback.format_exc())

    return result


@app.get("/", response_class=HTMLResponse)
def frontend():
    html_path = Path(__file__).parent / "frontend.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Frontend not found — make sure frontend.html is in the same folder.</h1>"
