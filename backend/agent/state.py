# agent/state.py
#
# WHY TypedDict?
# LangGraph requires state to be a TypedDict (or Pydantic model).
# TypedDict = Python dict, but with type hints so your IDE helps you.
# Every node receives this state, modifies some fields, returns it.
# LangGraph merges the returned dict back into the state automatically.
#
# WHY Optional everywhere?
# Pipeline start mein sirf csv_path hota hai, baaki sab None hote hain.
# Jaise jaise nodes run hote hain, fields fill hoti jaati hain.
# Optional = yeh field ab None hai, baad mein fill hogi.

from typing import TypedDict, Optional, Any


class AgentState(TypedDict):

    # ── INPUT ────────────────────────────────────────────────────────────────
    csv_path: str                    # Path to uploaded CSV file

    # ── NODE 1 output: Load & Validate ───────────────────────────────────────
    df: Optional[Any]                # Loaded DataFrame
                                     # Any kyunki TypedDict pd.DataFrame support nahi karta
    load_error: Optional[str]        # CSV load nahi hua toh error message yahan

    # ── NODE 2 output: EDA ───────────────────────────────────────────────────
    eda_summary: Optional[dict]      # shape, dtypes, nulls, describe(), correlations
    chart_base64s: Optional[list]    # [{"name": "Distributions", "data": "base64..."}]
                                     # WHY base64? File system pe save nahi karte —
                                     # deployment pe ephemeral filesystem hoti hai.
                                     # Base64 string directly JSON mein jaati hai.

    # ── NODE 3 output: Problem Type Detection ────────────────────────────────
    problem_type: Optional[str]      # "classification", "regression", ya "clustering"
    target_column: Optional[str]     # Kaunsa column predict karna hai (y)
    feature_columns: Optional[list]  # Kaunse columns features hain (X)

    # ── NODE 4 output: Feature Preparation ───────────────────────────────────
    X_train: Optional[Any]           # Training features (numpy array)
    X_test: Optional[Any]            # Test features (model ne kabhi nahi dekha)
    y_train: Optional[Any]           # Training labels/values
    y_test: Optional[Any]            # Test labels/values
    feature_names: Optional[list]    # Column names after OneHotEncoding (SHAP ke liye)

    # ── NODE 5 output: Model Selection ───────────────────────────────────────
    model_results: Optional[dict]    # {model_name: {metric: score}} sab models ka
    best_model_name: Optional[str]   # Winner model ka naam
    best_model: Optional[Any]        # Actual trained sklearn/XGBoost model object

    # ── NODE 6 output: Hyperparameter Tuning ─────────────────────────────────
    tuned_model: Optional[Any]       # GridSearchCV ke baad best model
    tuned_score: Optional[float]     # Final score on test set after tuning
    best_params: Optional[dict]      # Winning hyperparameter combination

    # ── NODE 7 output: Explanation ───────────────────────────────────────────
    shap_chart_b64: Optional[str]    # SHAP bar chart as base64 PNG string
    llm_explanation: Optional[str]   # Groq Llama ka human-readable explanation

    # ── FINAL ────────────────────────────────────────────────────────────────
    final_report: Optional[dict]     # Sab kuch ek JSON mein — API response
    error: Optional[str]             # Unexpected errors (load_error se alag)