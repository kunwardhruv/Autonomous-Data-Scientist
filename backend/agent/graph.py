# agent/graph.py
#
# WHY a separate graph.py?
# Nodes likhna alag kaam hai, unhe connect karna alag.
# graph.py = wiring file. Yahan decide hota hai:
#   - Kaunsa node kaunse ke baad aata hai (edges)
#   - Conditional edges: error hua toh pipeline rok do
#   - Final report kaise assemble hoga
#
# LangGraph ka flow:
# START → load_validate → (error?) → END
#                       → eda → detect_problem → prepare_features
#                         → model_selection → tuning → explain
#                         → assemble_report → END

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.node1_load import load_and_validate
from agent.nodes.node2_eda import run_eda
from agent.nodes.node3_problem_detection import detect_problem_type
from agent.nodes.node4_feature_prep import prepare_features
from agent.nodes.node5_model_selection import select_best_model
from agent.nodes.node6_tuning import tune_best_model
from agent.nodes.node7_explanation import explain_results


def assemble_report(state: AgentState) -> dict:
    """
    Sab nodes ke outputs ko ek clean JSON report mein bundle karo.
    FastAPI yeh report directly user ko return karega.

    WHY separate function?
    Har node sirf apna kaam karta hai — bundling ka kaam alag node mein.
    Clean separation of concerns.
    """
    return {
        "final_report": {
            "status": "success",
            "eda": {
                "shape": state["eda_summary"]["shape"],
                "missing_values": state["eda_summary"]["missing_values"],
                "numeric_columns": state["eda_summary"]["numeric_columns"],
                "categorical_columns": state["eda_summary"]["categorical_columns"],
                "charts": state.get("chart_base64s", []),   # base64 list [{name, data}]
            },
            "problem": {
                "type": state["problem_type"],
                "target": state["target_column"],
                "features": state["feature_columns"],
            },
            "model_selection": {
                "all_models": state["model_results"],
                "winner": state["best_model_name"],
            },
            "tuning": {
                "best_params": state["best_params"],
                "final_score": state["tuned_score"],
            },
            "explanation": {
                "shap_chart": state.get("shap_chart_b64"),   # base64 PNG string
                "llm_text": state["llm_explanation"],
            },
        }
    }


def check_load_error(state: AgentState) -> str:
    """
    Conditional edge function — Node 1 ke baad check karo.

    WHY conditional edges?
    LangGraph mein har edge ya toh fixed hoti hai ya conditional.
    Conditional = function run karo, output string ke basis par next node decide karo.
    "error" → END (pipeline rok do)
    "continue" → eda (aage badho)
    """
    if state.get("load_error"):
        print(f"\n  ✗ Load failed: {state['load_error']} — stopping pipeline")
        return "error"
    return "continue"


def build_graph():
    """
    LangGraph StateGraph banao with all 7 nodes + conditional error handling.
    Returns a compiled graph ready to invoke.
    """

    # ── Graph create karo with state schema ──────────────────────────────────
    graph = StateGraph(AgentState)

    # ── Nodes add karo ───────────────────────────────────────────────────────
    # add_node("naam", function) — function signature: (state) → dict
    graph.add_node("load_validate",   load_and_validate)
    graph.add_node("eda",             run_eda)
    graph.add_node("detect_problem",  detect_problem_type)
    graph.add_node("prepare_features", prepare_features)
    graph.add_node("model_selection", select_best_model)
    graph.add_node("tuning",          tune_best_model)
    graph.add_node("explain",         explain_results)
    graph.add_node("assemble_report", assemble_report)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("load_validate")

    # ── Conditional edge after Node 1 ────────────────────────────────────────
    # check_load_error return karta hai "error" ya "continue"
    # {return_value: next_node} mapping neeche hai
    graph.add_conditional_edges(
        "load_validate",
        check_load_error,
        {
            "error": END,        # Load fail → pipeline band
            "continue": "eda",   # Load success → EDA chalao
        }
    )

    # ── Fixed linear edges ────────────────────────────────────────────────────
    # Yeh nodes ek ke baad ek run karte hain — koi branching nahi
    graph.add_edge("eda",              "detect_problem")
    graph.add_edge("detect_problem",   "prepare_features")
    graph.add_edge("prepare_features", "model_selection")
    graph.add_edge("model_selection",  "tuning")
    graph.add_edge("tuning",           "explain")
    graph.add_edge("explain",          "assemble_report")
    graph.add_edge("assemble_report",  END)

    # ── Compile ───────────────────────────────────────────────────────────────
    # WHY compile? LangGraph graph validate karta hai (orphan nodes check,
    # cycle detection etc.) aur ek optimized runnable object return karta hai.
    return graph.compile()


# Singleton — ek baar build karo server start pe, har request pe reuse karo
# WHY? Graph build karna expensive hai — compilation + validation hota hai
agent_graph = build_graph()