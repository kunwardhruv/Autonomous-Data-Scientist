# api/server.py

import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from agent.graph import agent_graph
from agent.state import AgentState

app = FastAPI(
    title="Autonomous Data Scientist Agent",
    description="Upload a CSV → Get full EDA, model selection, and explanation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Autonomous Data Scientist Agent is running!"}


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    file_id = str(uuid.uuid4())
    csv_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"\n{'='*60}")
    print(f"New request: {file.filename} → {csv_path}")
    print(f"{'='*60}")

    try:
        initial_state: AgentState = {
            "csv_path": csv_path,
            "df": None, "load_error": None,
            "eda_summary": None, "chart_paths": None,
            "problem_type": None, "target_column": None, "feature_columns": None,
            "X_train": None, "X_test": None, "y_train": None, "y_test": None,
            "feature_names": None,
            "model_results": None, "best_model_name": None, "best_model": None,
            "tuned_model": None, "tuned_score": None, "best_params": None,
            "shap_chart_path": None, "llm_explanation": None,
            "final_report": None, "error": None,
        }

        final_state = agent_graph.invoke(initial_state)

        # final_report check karo — yahi real success indicator hai
        # load_error check hataya kyunki node4 edge case mein woh set hota tha
        # aur server galat 422 return karta tha even though pipeline complete thi
        report = final_state.get("final_report")

        if report is None:
            error_msg = (
                final_state.get("load_error")
                or final_state.get("error")
                or "Pipeline failed — check server logs."
            )
            return JSONResponse(
                status_code=422,
                content={"status": "error", "detail": error_msg}
            )

        return JSONResponse(content=report)

    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")

    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


@app.get("/chart/{filepath:path}")
def get_chart(filepath: str):
    filename = os.path.basename(filepath)
    chart_path = os.path.join("outputs/charts", filename)
    if not os.path.exists(chart_path):
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(chart_path, media_type="image/png")