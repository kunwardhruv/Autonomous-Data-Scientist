# agent/nodes/node5_model_selection.py

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from agent.state import AgentState


def select_best_model(state: AgentState) -> dict:
    print("\n[Node 5] Running model selection...")

    problem_type = state["problem_type"]
    X_train = state["X_train"]
    X_test  = state["X_test"]
    y_train = state["y_train"]
    y_test  = state["y_test"]

    # Node 4 ne data nahi diya (high cardinality edge case) — skip
    if X_train is None or y_train is None:
        print("  → Skipping model selection — no training data available.")
        return {"model_results": {}, "best_model_name": None, "best_model": None}

    if problem_type == "classification":
        candidates = _get_classifiers(y_train)
        results = _evaluate_classifiers(candidates, X_train, X_test, y_train, y_test)
        best_name = max(results, key=lambda k: results[k]["f1"])

    elif problem_type == "regression":
        candidates = _get_regressors()
        results = _evaluate_regressors(candidates, X_train, X_test, y_train, y_test)
        best_name = max(results, key=lambda k: results[k]["r2"])

    else:
        # Clustering
        return {"model_results": {}, "best_model_name": "KMeans", "best_model": None}

    best_model = candidates[best_name]

    print(f"\n  ── Model Comparison ──")
    for name, metrics in results.items():
        marker = "★" if name == best_name else " "
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"  {marker} {name:<30} {metrics_str}")
    print(f"\n  ✓ Winner: {best_name}")

    return {
        "model_results": results,
        "best_model_name": best_name,
        "best_model": best_model,
    }


def _get_classifiers(y_train) -> dict:
    n_classes = len(np.unique(y_train))
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            verbosity=0, random_state=42,
        ),
    }


def _get_regressors() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, verbosity=0, random_state=42),
    }


def _evaluate_classifiers(candidates, X_train, X_test, y_train, y_test) -> dict:
    results = {}
    average = "binary" if len(np.unique(y_train)) == 2 else "weighted"
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1": round(f1_score(y_test, y_pred, average=average, zero_division=0), 4),
            }
            print(f"  ✓ {name}: accuracy={results[name]['accuracy']}, f1={results[name]['f1']}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results[name] = {"accuracy": 0.0, "f1": 0.0}
    return results


def _evaluate_regressors(candidates, X_train, X_test, y_train, y_test) -> dict:
    results = {}
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results[name] = {
                "r2": round(r2_score(y_test, y_pred), 4),
                "rmse": round(np.sqrt(mse), 4),
            }
            print(f"  ✓ {name}: r2={results[name]['r2']}, rmse={results[name]['rmse']}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results[name] = {"r2": -99.0, "rmse": 99999.0}
    return results