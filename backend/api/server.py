# api/server.py
#
# FastAPI = modern Python web framework.
# Yeh kaam karta hai:
#   1. User CSV upload karta hai → POST /analyze
#   2. File temporarily save karo
#   3. LangGraph agent trigger karo
#   4. Report JSON mein return karo
#
# WHY no /chart/ endpoint?
# Pehle charts files mein save hoti thin aur /chart/ endpoint serve karta tha.
# Problem: deployment pe filesystem ephemeral hoti hai — restart pe files delete.
# Fix: Charts ab base64 strings hain, directly JSON response mein embedded.
# Frontend <img src="data:image/png;base64,..."> se render karta hai.

import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# .env file load karo (GROQ_API_KEY etc.)
load_dotenv()

from agent.graph import agent_graph
from agent.state import AgentState

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Autonomous Data Scientist Agent",
    description="Upload a CSV → Get full EDA, model selection, and explanation",
    version="1.0.0",
)

# CORS: Frontend (React localhost:5173) se requests allow karo
# WHY CORS? Browser security — cross-origin requests block hoti hain by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Production mein specific origin dena
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uploaded files temporary yahan store karo
# WHY temp? Analysis ke baad immediately delete karte hain — storage waste nahi
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.get("/")
def health_check():
    """Server alive hai? Railway/Render health checks ke liye."""
    return {"status": "ok", "message": "Autonomous Data Scientist Agent is running!"}


@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Main endpoint:
    1. CSV file receive karo
    2. Validate (CSV hai?)
    3. LangGraph agent run karo
    4. Report JSON mein return karo

    WHY async? FastAPI async endpoints concurrent requests handle kar sakta hai —
    ek user ka analysis chal raha ho toh doosra user wait nahi karta.
    """

    # ── File type validate karo ───────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    # ── File temporarily save karo ────────────────────────────────────────────
    # WHY uuid4? Random unique ID — multiple users ek saath upload kar sakein
    # bina filename collision ke
    file_id = str(uuid.uuid4())
    csv_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    with open(csv_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    print(f"\n{'='*60}")
    print(f"New request: {file.filename} → {csv_path}")
    print(f"{'='*60}")

    try:
        # ── Initial state banao — sirf csv_path, baaki sab None ──────────────
        initial_state: AgentState = {
            "csv_path": csv_path,
            "df": None, "load_error": None,
            "eda_summary": None, "chart_base64s": None,
            "problem_type": None, "target_column": None, "feature_columns": None,
            "X_train": None, "X_test": None, "y_train": None, "y_test": None,
            "feature_names": None,
            "model_results": None, "best_model_name": None, "best_model": None,
            "tuned_model": None, "tuned_score": None, "best_params": None,
            "shap_chart_b64": None, "llm_explanation": None,
            "final_report": None, "error": None,
        }

        # ── LangGraph agent invoke karo ───────────────────────────────────────
        # .invoke() = synchronously poora graph run karo
        # Returns final state after all nodes complete
        final_state = agent_graph.invoke(initial_state)

        # ── final_report check karo ───────────────────────────────────────────
        # WHY final_report check? Yahi real success indicator hai.
        # load_error check nahi karte kyunki node4 edge case mein
        # woh set hota tha aur server galat 422 return karta tha
        # even though pipeline complete ho jaati thi.
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
        # ── Cleanup: temp CSV delete karo ────────────────────────────────────
        # WHY finally? Exception ho ya na ho — file always delete hogi.
        # Storage waste nahi hoga.
        if os.path.exists(csv_path):
            os.remove(csv_path)