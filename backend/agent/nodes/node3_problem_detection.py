# agent/nodes/node3_problem_detection.py
#
# WHY auto-detect?
# User ne nahi bataya "classification hai ya regression" — agent ko khud samajhna hai.
# Logic simple hai:
#   - Last column = usually target (convention in most datasets)
#   - Target mein unique values <= 20 AND dtype int/object → Classification
#   - Target mein continuous float values → Regression
#   - Target column nahi dikh raha → Clustering

import pandas as pd
import numpy as np
from agent.state import AgentState

# Agar target column mein itne ya kam unique values hain → Classification
CLASSIFICATION_THRESHOLD = 20


def detect_problem_type(state: AgentState) -> dict:
    """
    Node 3: Data dekh ke automatically decide karo —
    Classification? Regression? Clustering?
    """
    print("\n[Node 3] Detecting problem type...")

    df: pd.DataFrame = state["df"]
    columns = list(df.columns)

    # ── Strategy: Last column ko target maano ────────────────────────────────
    # WHY last column? Most public datasets (Titanic, Iris, etc.) mein
    # target/label last hota hai. Yeh ek heuristic hai — production mein
    # user specify karta.
    target_col = columns[-1]
    feature_cols = columns[:-1]

    target_series = df[target_col]
    n_unique = target_series.nunique()
    dtype = target_series.dtype

    print(f"  → Candidate target: '{target_col}' | dtype: {dtype} | unique values: {n_unique}")

    # ── Decision logic ────────────────────────────────────────────────────────
    if dtype == "object" or dtype.name == "category":
        # String/category column → definitely Classification
        problem_type = "classification"
        reason = f"Target '{target_col}' is text/category type → Classification"

    elif n_unique <= CLASSIFICATION_THRESHOLD and dtype in [np.int64, np.int32, int]:
        # Integer with few unique values → Classification (e.g. 0/1, 1/2/3)
        problem_type = "classification"
        reason = f"Target '{target_col}' has {n_unique} unique int values → Classification"

    elif dtype in [np.float64, np.float32, float]:
        # Continuous float → Regression
        problem_type = "regression"
        reason = f"Target '{target_col}' is continuous float → Regression"

    elif n_unique > CLASSIFICATION_THRESHOLD:
        # Many unique int values → treat as Regression
        problem_type = "regression"
        reason = f"Target '{target_col}' has {n_unique} unique values → Regression"

    else:
        # Koi target identify nahi hua → Clustering (unsupervised)
        problem_type = "clustering"
        target_col = None      # Clustering mein target hota nahi
        feature_cols = columns
        reason = "No clear target column → Clustering"

    print(f"  ✓ Problem type: {problem_type}")
    print(f"  ✓ Reason: {reason}")
    print(f"  ✓ Features ({len(feature_cols)}): {feature_cols}")

    return {
        "problem_type": problem_type,
        "target_column": target_col,
        "feature_columns": feature_cols,
    }
