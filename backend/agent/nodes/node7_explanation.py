# agent/nodes/node7_explanation.py
#
# WHY explanation matters?
# Model accuracy jaanna kaafi nahi — "why did it predict this?" zaroori hai.
# SHAP (SHapley Additive exPlanations): game theory se aaya hai.
# Har feature ko credit/blame deta hai for each prediction.
# → "Passenger class aur fare ne survival predict karne mein sabse zyada contribution ki"

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from agent.state import AgentState

CHARTS_DIR = "outputs/charts"


def explain_results(state: AgentState) -> dict:
    """
    Node 7: SHAP plots + Groq LLM explanation generate karo.
    """
    print("\n[Node 7] Generating explanations...")

    tuned_model = state["tuned_model"]
    X_train = state["X_train"]
    X_test = state["X_test"]
    feature_names = state["feature_names"]
    problem_type = state["problem_type"]
    best_model_name = state["best_model_name"]
    eda_summary = state["eda_summary"]
    model_results = state["model_results"]
    tuned_score = state["tuned_score"]
    best_params = state["best_params"]
    target_col = state["target_column"]

    # ── SHAP Feature Importance ───────────────────────────────────────────────
    shap_chart_path = None
    shap_values_summary = {}

    if tuned_model is not None and problem_type != "clustering":
        shap_chart_path, shap_values_summary = _generate_shap_plot(
            tuned_model, X_test, feature_names, best_model_name
        )

    # ── LLM Explanation via Groq ──────────────────────────────────────────────
    llm_explanation = _generate_llm_explanation(
        eda_summary=eda_summary,
        problem_type=problem_type,
        target_col=target_col,
        model_results=model_results,
        best_model_name=best_model_name,
        tuned_score=tuned_score,
        best_params=best_params,
        shap_summary=shap_values_summary,
    )

    return {
        "shap_chart_path": shap_chart_path,
        "llm_explanation": llm_explanation,
    }


def _generate_shap_plot(model, X_test, feature_names, model_name) -> tuple:
    """
    SHAP values calculate karo aur summary bar plot save karo.
    Returns (chart_path, top_features_dict).
    """
    print("  → Computing SHAP values...")

    try:
        # ── Choose right SHAP Explainer ───────────────────────────────────────
        # TreeExplainer: tree-based models (RF, XGBoost, DT) ke liye fast
        # KernelExplainer: any model but slow — fallback
        tree_models = ["Random Forest", "XGBoost", "Decision Tree"]

        if any(name in model_name for name in tree_models):
            explainer = shap.TreeExplainer(model)
            # Use max 500 samples (speed ke liye)
            X_sample = X_test[:500]
            shap_values = explainer.shap_values(X_sample)
        else:
            # KernelExplainer — slow, use very small sample
            X_sample = shap.sample(X_test, 50)   # Only 50 samples
            explainer = shap.KernelExplainer(model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)

        # ── Handle multi-class: use first class SHAP values ──────────────────
        # For binary classification, shap_values is [negative_class, positive_class]
        if isinstance(shap_values, list):
            # Multi-class ya binary — last class ki values lo (positive class)
            sv = shap_values[-1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        # ── Mean absolute SHAP values = feature importance ───────────────────
        mean_abs_shap = np.abs(sv).mean(axis=0)

        # Top features dict (for LLM context)
        if feature_names and len(feature_names) == len(mean_abs_shap):
            sorted_idx = np.argsort(mean_abs_shap)[::-1]
            top_n = min(10, len(feature_names))
            shap_summary = {
                feature_names[i]: round(float(mean_abs_shap[i]), 4)
                for i in sorted_idx[:top_n]
            }
        else:
            shap_summary = {}

        # ── Plot: Horizontal bar chart of mean |SHAP| values ─────────────────
        top_k = min(15, len(mean_abs_shap))
        sorted_idx = np.argsort(mean_abs_shap)[-top_k:]

        fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.45)))

        y_pos = range(top_k)
        values = mean_abs_shap[sorted_idx]
        labels = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in sorted_idx]

        bars = ax.barh(y_pos, values, align="center", color="#5B8DB8", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Mean |SHAP Value| (average impact on model output)")
        ax.set_title(f"Feature Importance (SHAP)\n{model_name}", fontsize=12)

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", ha="left", fontsize=8)

        path = os.path.join(CHARTS_DIR, "shap_importance.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"  ✓ SHAP chart saved: {path}")
        print(f"  ✓ Top feature: {list(shap_summary.items())[0] if shap_summary else 'N/A'}")

        return path, shap_summary

    except Exception as e:
        print(f"  ✗ SHAP failed: {e} — skipping")
        return None, {}


def _generate_llm_explanation(
    eda_summary, problem_type, target_col,
    model_results, best_model_name, tuned_score,
    best_params, shap_summary
) -> str:
    """
    Groq Llama ko context do, human-readable explanation generate karo.
    """
    print("  → Calling Groq LLM for explanation...")

    try:
        # Lazy import — Groq available hai tabhi call karo
        from langchain_groq import ChatGroq
        import os

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return "⚠️ GROQ_API_KEY not set. Set it in .env to get LLM explanations."

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,   # Low temp = consistent, factual output
            api_key=groq_api_key,
        )

        # ── Build context for LLM ──────────────────────────────────────────────
        shape = eda_summary.get("shape", {})
        missing = eda_summary.get("missing_values", {})
        model_table = "\n".join([
            f"  - {name}: {metrics}"
            for name, metrics in model_results.items()
        ])
        top_features = "\n".join([
            f"  {i+1}. {feat} (SHAP={val})"
            for i, (feat, val) in enumerate(list(shap_summary.items())[:5])
        ]) if shap_summary else "  (Not available)"

        prompt = f"""You are a data scientist explaining ML results to a non-technical person.
Be clear, concise, and insightful. Use simple language. No jargon without explanation.

=== DATASET OVERVIEW ===
- Rows: {shape.get('rows', '?')}, Columns: {shape.get('columns', '?')}
- Task: {problem_type.upper()} — predicting '{target_col}'
- Missing values: {list(missing.keys()) if missing else 'None'}

=== MODEL COMPARISON ===
{model_table}

=== WINNER: {best_model_name} ===
- Final score: {tuned_score}
- Best hyperparameters: {best_params}

=== TOP IMPORTANT FEATURES (SHAP) ===
{top_features}

Please provide:
1. A brief dataset description (2-3 sentences)
2. Why {best_model_name} likely performed best for this task
3. What the top features mean — what drove the predictions?
4. One actionable insight from this analysis
5. Any data quality concerns to watch out for

Keep the tone friendly and educational. Max 300 words."""

        response = llm.invoke(prompt)
        explanation = response.content

        print(f"  ✓ LLM explanation generated ({len(explanation)} chars)")
        return explanation

    except Exception as e:
        print(f"  ✗ LLM explanation failed: {e}")
        return f"Could not generate LLM explanation: {str(e)}"
