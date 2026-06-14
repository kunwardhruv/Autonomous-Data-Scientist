# agent/graph.py
#
# WHY a separate graph.py?
# Nodes likhna alag kaam hai, unhe connect karna alag.
# graph.py = wiring file. Yahan decide hota hai:
#   - Kaunsa node kaunse ke baad aata hai
#   - Conditional edges: error hua toh pipeline rok do

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.node1_load import load_and_validate
from agent.nodes.node2_eda import run_eda
from agent.nodes.node3_problem_detection import detect_problem_type
from agent.nodes.node4_feature_prep import prepare_features
from agent.nodes.node5_model_selection import select_best_model
from agent.nodes.node6_tuning import tune_best_model
from agent.nodes.node7_explanation import explain_results


# ── Final report assembler (not a LangGraph node, just a helper) ─────────────
def assemble_report(state: AgentState) -> dict:
    """
    Sab nodes ke output ko ek clean JSON report mein bundle karo.
    FastAPI yeh report user ko return karega.
    """
    return {
        "final_report": {
            "status": "success",
            "eda": {
                "shape": state["eda_summary"]["shape"],
                "missing_values": state["eda_summary"]["missing_values"],
                "numeric_columns": state["eda_summary"]["numeric_columns"],
                "categorical_columns": state["eda_summary"]["categorical_columns"],
                "charts": state["chart_paths"],
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
                "shap_chart": state["shap_chart_path"],
                "llm_text": state["llm_explanation"],
            },
        }
    }


# ── Conditional edge function ─────────────────────────────────────────────────
def check_load_error(state: AgentState) -> str:
    """
    Node 1 ke baad check karo — load error tha?
    Agar haan → pipeline rok do (END).
    Agar nahi → EDA continue karo.

    WHY conditional edges?
    LangGraph mein har edge ya toh fixed hoti hai ya conditional.
    Conditional = function run karo, output string ke basis par next node decide karo.
    """
    if state.get("load_error"):
        print(f"\n  ✗ Load failed: {state['load_error']} — stopping pipeline")
        return "error"
    return "continue"


# ════════════════════════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ════════════════════════════════════════════════════════════════════════════

def build_graph():
    """
    LangGraph StateGraph banao with all 7 nodes + conditional error handling.
    Returns a compiled graph ready to invoke.
    """

    # ── Create the graph with our state schema ────────────────────────────────
    graph = StateGraph(AgentState)

    # ── Add all nodes ─────────────────────────────────────────────────────────
    # add_node("name", function) — function = (state) → dict
    graph.add_node("load_validate", load_and_validate)
    graph.add_node("eda", run_eda)
    graph.add_node("detect_problem", detect_problem_type)
    graph.add_node("prepare_features", prepare_features)
    graph.add_node("model_selection", select_best_model)
    graph.add_node("tuning", tune_best_model)
    graph.add_node("explain", explain_results)
    graph.add_node("assemble_report", assemble_report)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("load_validate")

    # ── Conditional edge after Node 1 ────────────────────────────────────────
    # If error → END, else → eda
    graph.add_conditional_edges(
        "load_validate",                  # Source node
        check_load_error,                 # Function that returns a string
        {
            "error": END,                 # "error" string → go to END
            "continue": "eda",            # "continue" string → go to eda
        }
    )

    # ── Fixed edges (linear pipeline) ────────────────────────────────────────
    graph.add_edge("eda", "detect_problem")
    graph.add_edge("detect_problem", "prepare_features")
    graph.add_edge("prepare_features", "model_selection")
    graph.add_edge("model_selection", "tuning")
    graph.add_edge("tuning", "explain")
    graph.add_edge("explain", "assemble_report")
    graph.add_edge("assemble_report", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    # WHY compile? LangGraph validates the graph (no orphan nodes, no cycles, etc.)
    # and returns an optimized runnable object.
    return graph.compile()


# Singleton — ek baar build karo, reuse karo
agent_graph = build_graph()
