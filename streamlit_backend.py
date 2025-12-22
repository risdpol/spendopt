# main.py
import pandas as pd
import pickle
from langgraph.graph import StateGraph, END

from data_preparation import (
    summarize_for_budget,
    load_and_filter_data,
    BudgetSuggestionNode,
    ConstraintSelectionNode,
    FinalDataLayerNode
)
from model_building import ModelBuildingNode
from spend_simulation import SimulationNode

# ---------------- CONFIG ---------------- #

GROQ_API_KEY = "YOUR_KEY"

FILTERS = {
    "region": "region_1",
    "channel": "retail",
    "customer": "customer_1",
    "category": "CAT_1",
    "brand": "brand_1",
    "sku_id": None,
    "store_id": None
}

# ---------------- GRAPH BUILDERS ---------------- #

def build_budget_graph(api_key):
    g = StateGraph(dict)
    g.add_node("budget", BudgetSuggestionNode(api_key))
    g.set_entry_point("budget")
    return g.compile()

def build_constraints_graph(api_key):
    g = StateGraph(dict)
    g.add_node("constraints", ConstraintSelectionNode(api_key))
    g.add_node("final_data", FinalDataLayerNode())
    g.set_entry_point("constraints")
    g.add_edge("constraints", "final_data")
    g.add_edge("final_data", END)
    return g.compile()

def build_model_graph():
    g = StateGraph(dict)
    g.add_node("model_layer", ModelBuildingNode())
    g.set_entry_point("model_layer")
    g.add_edge("model_layer", END)
    return g.compile()

def build_simulation_graph():
    g = StateGraph(dict)
    g.add_node("simulation", SimulationNode())
    g.set_entry_point("simulation")
    g.add_edge("simulation", END)
    return g.compile()

# ---------------- MAIN PIPELINE ---------------- #

def run_pipeline(
    data_path: str,
    objective: str,
    roi_min: float,
    promo_share: dict,
    horizon: int,
    budget_override: float | None = None
):
    # -------- Load & Filter Data -------- #
    df = pd.read_excel(data_path, sheet_name="Sheet1")
    filtered_df = load_and_filter_data(df, FILTERS)

    # -------- STAGE 1: Budget -------- #
    budget_summary = summarize_for_budget(filtered_df)

    budget_graph = build_budget_graph(GROQ_API_KEY)
    stage1_output = budget_graph.invoke({
        "budget_summary": budget_summary
    })

    budget = budget_override or stage1_output["budget"]

    # -------- STAGE 2: Constraints -------- #
    constraints_graph = build_constraints_graph(GROQ_API_KEY)

    stage2_state = {
        "budget": budget,
        "budget_details": stage1_output["budget_details"],
        "user_constraints": {
            "objective": objective,
            "roi_min": roi_min,
            "promo_share": promo_share,
            "horizon": horizon,
        },
        "filters": FILTERS,
    }

    final_state = constraints_graph.invoke(stage2_state)

    # -------- STAGE 3: Model Building -------- #
    model_graph = build_model_graph()

    final_data = final_state["final_data_layer"]

    # 🔑 Model expects this key
    final_data["HORIZON_MONTHS"] = final_data["horizon"]

    output = model_graph.invoke({
        "filtered_df": filtered_df,
        **final_data,
    })

    # -------- Persist artifacts -------- #
    pd.DataFrame(output["plan"]).to_excel(
        "optimized_plan.xlsx",
        index=False
    )

    with open("model_store.pkl", "wb") as f:
        pickle.dump(output["model_store"], f)

    # -------- RETURN (NO SIMULATION HERE) -------- #
    return {
        "budget_details": stage1_output["budget_details"],
        "final_data": final_data,
        "summary": output["summary"],
        "plan": output["plan"],
        "model_store": output["model_store"],
        "ctx": output["ctx"],
    }
