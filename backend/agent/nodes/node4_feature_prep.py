# agent/nodes/node4_feature_prep.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from agent.state import AgentState

MAX_CATEGORIES = 50


def prepare_features(state: AgentState) -> dict:
    print("\n[Node 4] Preparing features...")

    df: pd.DataFrame = state["df"]
    problem_type: str = state["problem_type"]
    target_col: str = state["target_column"]
    feature_cols: list = state["feature_columns"]

    # ════════════════════════════════════════════════════════════════════════
    # CLUSTERING
    # ════════════════════════════════════════════════════════════════════════
    if problem_type == "clustering":
        X = df[feature_cols].copy()
        X = _basic_clean(X)
        X_array = StandardScaler().fit_transform(X.select_dtypes(include=[np.number]))
        print(f"  ✓ Clustering prep done. Shape: {X_array.shape}")
        return {
            "X_train": X_array, "X_test": None,
            "y_train": None, "y_test": None,
            "feature_names": list(X.select_dtypes(include=[np.number]).columns),
        }

    # ════════════════════════════════════════════════════════════════════════
    # CLASSIFICATION / REGRESSION
    # ════════════════════════════════════════════════════════════════════════
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows jahan target NaN hai
    nan_mask = pd.isnull(y)
    if nan_mask.any():
        print(f"  → Dropping {nan_mask.sum()} rows with NaN in target")
        X = X[~nan_mask]
        y = y[~nan_mask]

    # Target encode karo
    if problem_type == "classification" and y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"  → Target encoded: {list(le.classes_)} → {list(range(len(le.classes_)))}")
    else:
        y = y.values

    # Column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"  → Numeric features: {numeric_cols}")
    print(f"  → Categorical features: {categorical_cols}")

    # High cardinality columns drop karo (RAM explosion rokne ke liye)
    safe_categorical = []
    dropped_cols = []
    for col in categorical_cols:
        n_unique = X[col].nunique()
        if n_unique <= MAX_CATEGORIES:
            safe_categorical.append(col)
        else:
            dropped_cols.append(col)
            print(f"  → Dropping '{col}' — {n_unique} unique values (too high for OHE)")

    if dropped_cols:
        X = X.drop(columns=dropped_cols)
        print(f"  → Total dropped: {dropped_cols}")

    categorical_cols = safe_categorical

    # Edge case — koi feature nahi bacha
    # WHY 'error' key? 'load_error' use karte toh server 422 return karta
    # chahe pipeline complete ho jaaye. 'error' key sirf log ke liye hai.
    if not numeric_cols and not categorical_cols:
        print("  ✗ No usable features remaining after dropping high-cardinality columns.")
        return {
            "error": "No usable feature columns after dropping high-cardinality columns.",
            "X_train": None, "X_test": None,
            "y_train": None, "y_test": None,
            "feature_names": [],
        }

    # ColumnTransformer
    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # Train/Test split
    stratify = y if (problem_type == "classification" and len(np.unique(y)) > 1) else None

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feature_names = _get_feature_names(preprocessor, numeric_cols, categorical_cols)

    print(f"  ✓ X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  ✓ y_train distribution: {dict(zip(*np.unique(y_train, return_counts=True))) if problem_type == 'classification' else 'continuous'}")

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
    }


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    return df


def _get_feature_names(preprocessor, numeric_cols, categorical_cols) -> list:
    names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"]["encoder"]
        names.extend(ohe.get_feature_names_out(categorical_cols).tolist())
    return names