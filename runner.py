# main.py
import pandas as pd
import pickle
from langgraph.graph import StateGraph, END
from data_preparation import (
    summarize_for_budget,
    get_user_constraints,
    load_and_filter_data,
    BudgetSuggestionNode,
    ConstraintSelectionNode,
    FinalDataLayerNode
)
from model_building import ModelBuildingNode
from spend_simulation import SimulationNode

GROQ_API_KEY = "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"

FILTERS = {
    "region": "region_1",
    "channel": "retail",
    "customer": "customer_1",
    "category": "CAT_1",
    "brand": "brand_1",
    "sku_id": None,
    "store_id": None
}

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

def main():
    # data_path_local = r'C:\Users\shama.perween\Shama\LLM Powered Spend Optimization\Inputs\Data_Optimisation.xlsx'
    data_path_databricks = r"C:\Users\rishabh.goyal\Downloads\LLM-Powered-Spend-Optimization\LLM-Powered-Spend-Optimization\Inputs\Data_Optimisation.xlsx"
    # Load dataset
    df = pd.read_excel(
        data_path_databricks,
        sheet_name="Sheet1"
    )

    # ---- FILTER DATA ----
    filtered_df = load_and_filter_data(df, FILTERS)

    # ---- BUILD BUDGET SUMMARY ----
    budget_summary = summarize_for_budget(filtered_df)

    # STAGE 1 — Budget Suggestion
    # Initial state SHOULD contain ONLY budget summary
    stage1_state = {
        "budget_summary": budget_summary
    }

    budget_graph = build_budget_graph(GROQ_API_KEY)
    stage1_output = budget_graph.invoke(stage1_state)
    print("\n===== BUDGET SUGGESTION =====")
    print(stage1_output["budget_details"])

    # ---- Ask user ----
    if input("\nAre you satisfied with the user budgete? (y/n): ").lower() == "n":
        min_budget = budget_summary["min_spend"]
        max_budget = budget_summary["max_spend"]
        print(f"\nRecommended budget range: {min_budget} - {max_budget}")
        new_budget = float(input("Enter your desired budget: "))
        stage1_output["budget"] = new_budget
        stage1_output["budget_details"]["recommended_budget"] = new_budget

    # STAGE 2 — Constraint + Final Data
    # ---- GET USER INPUTS (objective, ROI, promo share, horizon) ----

    user_constraints = get_user_constraints()
    stage2_state = {
    "budget": stage1_output["budget"],
    "budget_details": stage1_output["budget_details"],
    "user_constraints": user_constraints,
    "filters": FILTERS
}

    constraints_graph = build_constraints_graph(GROQ_API_KEY)
    final_state = constraints_graph.invoke(stage2_state)

    print("\n===== FINAL DATA LAYER =====")
    print(final_state["final_data_layer"])

    # STAGE 3 — Model Building

    model_graph = build_model_graph()
    state_for_model = {
    "filtered_df": filtered_df,
    "budget": final_state["final_data_layer"]["budget"],
    "objective": final_state["final_data_layer"]["objective"],
    "roi_min": final_state["final_data_layer"]["roi_min"],
    "promo_share": final_state["final_data_layer"]["promo_share"],
    "HORIZON_MONTHS": final_state["final_data_layer"]["horizon"],
    "filters": final_state["final_data_layer"]["filters"]
}

    # state_for_model = {
    #     "filtered_df": filtered_df,    # from data_preparation
    #     "final_data_layer": final_state["final_data_layer"]
    # }

    output = model_graph.invoke(state_for_model)

    plan = output["plan"]
    plan_df = pd.DataFrame(plan)
    plan_df.to_excel("optimized_plan.xlsx", index=False)

    summary = output["summary"]
    print(f"Summary of Optimization:\n{summary}")

    model_store = output["model_store"]
    with open("model_store.pkl", "wb") as f:
        pickle.dump(model_store, f)

    # STAGE 4 — Spend Simulation / Re-optimization

    simulation_graph = build_simulation_graph()
    sim_state = {
        "plan": output["plan"],
        "model_store": output["model_store"],
        "ctx": output["ctx"]
    }

    sim_out = simulation_graph.invoke(sim_state)
    
    updated_plan = sim_out["updated_plan"]
    print("\n===== UPDATED PLAN AFTER SIMULATION =====")
    new_plan_df = pd.DataFrame(updated_plan)
    print(new_plan_df.head(10))
    new_plan_df.to_excel("updated_optimized_plan.xlsx", index=False)

    return final_state["final_data_layer"]

if __name__ == "__main__":
    main()
