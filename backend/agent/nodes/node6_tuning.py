# agent/nodes/node6_tuning.py

from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
import pandas as pd

from agent.state import AgentState

CLASSIFICATION_PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"],
    },
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "Decision Tree": {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
}

REGRESSION_PARAM_GRIDS = {
    "Ridge Regression": {"alpha": [0.01, 0.1, 1.0, 10.0]},
    "Random Forest": {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
    },
    "XGBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
    "Decision Tree": {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    },
}


def tune_best_model(state: AgentState) -> dict:
    print("\n[Node 6] Tuning hyperparameters...")

    problem_type    = state["problem_type"]
    best_model      = state["best_model"]
    best_model_name = state["best_model_name"]
    X_train         = state["X_train"]
    y_train         = state["y_train"]
    X_test          = state["X_test"]
    y_test          = state["y_test"]

    # Koi bhi None ho → skip
    if any(v is None for v in [best_model, X_train, y_train]):
        print("  → Skipping tuning — missing model or data.")
        return {"tuned_model": best_model, "tuned_score": 0.0, "best_params": {}}

    if problem_type == "clustering":
        print("  → Skipping tuning for clustering.")
        return {"tuned_model": best_model, "tuned_score": 0.0, "best_params": {}}

    param_grids = (
        CLASSIFICATION_PARAM_GRIDS
        if problem_type == "classification"
        else REGRESSION_PARAM_GRIDS
    )
    param_grid = param_grids.get(best_model_name, None)

    if param_grid is None:
        print(f"  → No param grid for {best_model_name}, using default.")
        try:
            score = cross_val_score(best_model, X_train, y_train, cv=3).mean()
        except Exception:
            score = 0.0
        return {
            "tuned_model": best_model,
            "tuned_score": round(float(score), 4),
            "best_params": {},
        }

    scoring = "f1_weighted" if problem_type == "classification" else "r2"

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"  → Grid searching {best_model_name} — {n_combos} combinations × 5 folds...")

    try:
        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=0,
            error_score=0.0,
        )
        grid_search.fit(X_train, y_train)
        tuned_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

    except Exception as e:
        print(f"  ✗ GridSearch failed: {e} — using untuned model")
        return {
            "tuned_model": best_model,
            "tuned_score": 0.0,
            "best_params": {},
        }

    # Final score
    try:
        if X_test is None or y_test is None:
            tuned_score = float(grid_search.best_score_)
        elif problem_type == "classification":
            from sklearn.metrics import f1_score
            y_pred = tuned_model.predict(X_test)
            avg = "binary" if len(np.unique(y_train)) == 2 else "weighted"
            tuned_score = f1_score(y_test, y_pred, average=avg, zero_division=0)
        else:
            from sklearn.metrics import r2_score
            y_pred = tuned_model.predict(X_test)
            tuned_score = r2_score(y_test, y_pred)
    except Exception as e:
        print(f"  ✗ Score calculation failed: {e} — using CV score")
        tuned_score = float(grid_search.best_score_)

    print(f"  ✓ Best params: {best_params}")
    print(f"  ✓ Final score: {tuned_score:.4f}")

    return {
        "tuned_model": tuned_model,
        "tuned_score": round(float(tuned_score), 4),
        "best_params": best_params,
    }