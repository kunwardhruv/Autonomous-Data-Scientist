# agent/state.py
#
# WHY TypedDict?
# LangGraph requires state to be a TypedDict (or Pydantic model).
# TypedDict = Python dict, but with type hints so your IDE helps you.
# Every node receives this state, modifies some fields, returns it.
# LangGraph merges the returned dict back into the state automatically.

from typing import TypedDict, Optional, Any
import pandas as pd


class AgentState(TypedDict):
    # ── INPUT ────────────────────────────────────────────────────────────────
    csv_path: str                    # Path to uploaded CSV file

    # ── NODE 1 output: Load & Validate ───────────────────────────────────────
    df: Optional[Any]                # The loaded DataFrame (pd.DataFrame)
                                     # typed as Any because TypedDict doesn't
                                     # support pd.DataFrame directly
    load_error: Optional[str]        # If CSV couldn't be loaded, error goes here

    # ── NODE 2 output: EDA ───────────────────────────────────────────────────
    eda_summary: Optional[dict]      # All stats: shape, dtypes, nulls, describe()
    chart_paths: Optional[list]      # List of saved chart image paths

    # ── NODE 3 output: Problem Type Detection ────────────────────────────────
    problem_type: Optional[str]      # "classification", "regression", or "clustering"
    target_column: Optional[str]     # Which column is the target (y)?
    feature_columns: Optional[list]  # Which columns are features (X)?

    # ── NODE 4 output: Feature Preparation ───────────────────────────────────
    X_train: Optional[Any]           # Training features (numpy array)
    X_test: Optional[Any]            # Test features
    y_train: Optional[Any]           # Training labels/values
    y_test: Optional[Any]            # Test labels/values
    feature_names: Optional[list]    # Column names after encoding (for SHAP later)

    # ── NODE 5 output: Model Selection ───────────────────────────────────────
    model_results: Optional[dict]    # {model_name: {metric: score}} for all models
    best_model_name: Optional[str]   # Name of the winner
    best_model: Optional[Any]        # The actual trained model object

    # ── NODE 6 output: Hyperparameter Tuning ─────────────────────────────────
    tuned_model: Optional[Any]       # Tuned model object
    tuned_score: Optional[float]     # Best score after tuning
    best_params: Optional[dict]      # The winning hyperparameters

    # ── NODE 7 output: Explanation ───────────────────────────────────────────
    shap_chart_path: Optional[str]   # Path to SHAP feature importance chart
    llm_explanation: Optional[str]   # Groq's text explanation of results

    # ── FINAL ────────────────────────────────────────────────────────────────
    final_report: Optional[dict]     # Everything bundled for the API response
    error: Optional[str]             # Any unexpected error during pipeline
