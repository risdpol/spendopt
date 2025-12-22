# data_preparation.py
import json
import groq
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, END
from pydantic import BaseModel
from typing import Dict, Optional
from llm_client import create_client
client = create_client()


def extract_json(text: str):
    """
    Extract the first valid JSON object from an LLM response.
    Handles:
    - Leading/trailing text
    - Code fences ```json ... ```
    - Pretty printed JSON
    """
    text = text.strip()

    # Remove code fences
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```python", "")
        text = text.replace("```", "")
        text = text.strip()

    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM output:\n" + text)

    json_str = text[start:end+1]
    return json.loads(json_str)


# ---------------------------------------------------------
# USER INPUT COLLECTION FUNCTION
# ---------------------------------------------------------

def get_user_constraints():
    """
    Collect ROI_MIN, OBJECTIVE, PROMO_SHARE from the user.
    Budget will come from LLM, NOT USER.
    """

    print("\nSelect Objective:")
    print("1. revenue")
    print("2. profit")
    print("3. roi")
    print("4. npm")

    obj_map = {1: "revenue", 2: "profit", 3: "roi", 4: "npm"}
    obj_choice = int(input("Enter choice (1-4): ").strip())
    objective = obj_map.get(obj_choice, "revenue")

    roi_min = float(input("Enter ROI_MIN value (e.g., 0.25 for 25%): ").strip())

    # print("\nEnter Promo Share in JSON format:")
    # print('Example: {"discount":0.4, "display":0.3, "bogo":0.2, "feature":0.1}')
    # promo_share = eval(input("Promo Share Dict: ").strip())

    print("Enter promo share for the following promos (values will be auto-normalized): ")

    discount_share = float(input("Enter promo share for discount: "))
    display_share  = float(input("Enter promo share for display: "))
    bogo_share     = float(input("Enter promo share for bogo: "))
    feature_share  = float(input("Enter promo share for feature: "))

    # Create dictionary with raw values
    PROMO_SHARE = {
        "discount": discount_share,
        "display":  display_share,
        "bogo":     bogo_share,
        "feature":  feature_share
    }

    # Calculate total
    total_share = sum(PROMO_SHARE.values())
    print(f"\nOriginal total share = {total_share}")

    # Auto-normalize if not equal to 1
    if abs(total_share - 1) > 1e-6:
        print("[INFO] Auto-normalizing promo shares...")

        for k in PROMO_SHARE:
            PROMO_SHARE[k] = PROMO_SHARE[k] / total_share

        print(f"New normalized total = {sum(PROMO_SHARE.values())}")
    else:
        print("[INFO] Shares already sum to 1. No normalization needed.")

    print("\nFinal PROMO_SHARE dictionary:")
    print(PROMO_SHARE)

    horizon = int(input("\nEnter optimization horizon in months (e.g., 3, 6, 9, 12): ").strip())
    return {
        "objective": objective,
        "roi_min": roi_min,
        "promo_share": PROMO_SHARE,
        "horizon": horizon
    }


# ------------------------------------------------
#  LOAD & FILTER DATA (original cell 0)
# ------------------------------------------------

def load_and_filter_data(df: pd.DataFrame, FILTERS: dict):
    """
    Original 'LOAD & MAP COLUMNS' cell wrapped into a function.
    Uses and updates global df and BUDGET_GLOBAL exactly as before.
    """

    print("[INFO] Step 0: Loading raw data and applying filters...")

    expected_raw = [
        "store_id","sku_id","sku_id","category","region","channel",
        "week", "year",
        "promo_spends", "sales_units", "total_cogs",
        "promo_mechanics", "roi_pct", "incremental_revenue", "net_profit_margin", "total_revenue"
    ]
    missing_raw = [c for c in expected_raw if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required columns in RGM_RAW_DATA.xlsx: {missing_raw}")

    df['promo_mechanics'] = df['promo_mechanics'].replace({"none": np.nan})
    df["week"] = df["week"].astype(int)
    df["year"] = df["year"].astype(int)
    df["promo_spends"] = df["promo_spends"].astype(float)
    df["total_revenue"] = df["total_revenue"].astype(float)
    df["sales_units"] = df["sales_units"].astype(float)
    df["roi_pct"] = df["roi_pct"].astype(float)

    # Apply FILTERS
    if any(v is not None for v in FILTERS.values()):
        mask = pd.Series(True, index=df.index)

        for col, allowed in FILTERS.items():
            if allowed is None:
                continue
            if col not in df.columns:
                print(f"[WARN] Column '{col}' not in df.columns, skipping this filter.")
                continue

            if isinstance(allowed, (list, tuple, set, pd.Index, np.ndarray)):
                allowed_list = [str(v) for v in allowed]
            else:
                allowed_list = [str(allowed)]

            col_as_str = df[col].astype(str)

            before = mask.sum()
            mask &= col_as_str.isin(allowed_list)
            after = mask.sum()
            print(f"[DEBUG] Filter on '{col}' with {allowed_list}: {before} -> {after} rows")

        df = df[mask].copy()
        print(f"[INFO] Rows after all FILTERS: {len(df)}")

        if df.empty:
            raise RuntimeError("No rows left after applying FILTERS; please relax FILTERS or check values/types.")

    print(f"[INFO] Step 0 complete. ")
    print("\n----------------------------------------------------------------------------")
    return df

# ---------------------------------------------------------
# SUMMARIZER FOR BUDGET SUGGESTION
# ---------------------------------------------------------

def summarize_for_budget(df: pd.DataFrame):
    """Small summary passed to LLM for budget suggestion."""
    last_year = int(df["year"].max())
    last_year_spend = float(df[df["year"] == last_year]["promo_spends"].sum())
    print(f"Last year total spend = {last_year_spend:.2f}")

    # Allowed range = 80% to 120%
    min_budget = 0.80 * last_year_spend
    max_budget = 1.20 * last_year_spend
    return {
        "min_spend": min_budget,
        "max_spend": max_budget,    
        "median_spend": float(df["promo_spends"].median()),
        "last_year_spend": last_year_spend,
        "num_skus": df["sku_id"].nunique(),
        "num_weeks": df["week"].nunique(),
        "promo_summary": (
            df.groupby("promo_mechanics")["promo_spends"]
              .mean()
              .round(2)
              .to_dict()
        ),
    }


# ---------------------------------------------------------
# BUDGET SUGGESTION NODE (LLM)
# ---------------------------------------------------------

class BudgetSuggestion(BaseModel):
    recommended_budget: float
    lower_limit: float
    upper_limit: float
    explanation: str

class BudgetSuggestionNode:
    def __init__(self, api_key: str):
        # self.client = groq.Groq(api_key=api_key)
        self.client = client

    def __call__(self, state: dict):
        summary = state["budget_summary"]
        prompt = f"""
                    You are a strict JSON generator.
                    Your task:
                    - Suggest a promotional marketing budget between 80%–120% of last year's spend.
                    - Use the dataset summary below.
                    - Use the given objective.
                    - STRICTLY return ONLY a valid JSON object.
                    - NO explanation outside the JSON.
                    - NO markdown.
                    - NO code fences.
                    - NO extra text.

                    Dataset Summary:
                    {summary}

                    Return JSON ONLY in this EXACT format (values filled in):
                    {{
                    "recommended_budget": <float>,
                    "lower_limit": <float>,
                    "upper_limit": <float>,
                    "explanation": "<string>"
                    }}
"""

        response = self.client.beta.chat.completions.parse(
            # model="llama-3.1-8b-instant",
            model = "gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format=BudgetSuggestion
        )

        raw = response.choices[0].message.content.strip()
        print(f'[DEBUG] Raw LLM output for budget suggestion:\n{raw}\n')

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            #  fallback
            print("[WARN] JSONDecodeError encountered, using fallback extraction.")
            parsed = extract_json(raw)
            
        # state["budget_details"] = extract_json(raw)
        state["budget_details"] = parsed
        state["budget"] = state["budget_details"]["recommended_budget"]
        print(f"[DEBUG] Parsed LLLM Output - Suggested Budget: {state['budget']}")
        print("\n----------------------------------------------------------------------------")

        # parsed = BudgetSuggestion.parse_raw(response.choices[0].message.content)
        # state["budget"] = parsed.recommended_budget
        # state["budget_details"] = parsed.dict()
        return state


# ---------------------------------------------------------
# CONSTRAINT SELECTION NODE (LLM VALIDATION)
# ---------------------------------------------------------

class ConstraintOutput(BaseModel):
    selected_objective: str
    roi_min: float
    promo_share: Optional[Dict[str, float]] = None
    horizon: int
    explanation: str

class ConstraintSelectionNode:
    def __init__(self, api_key: str):
        # self.client = groq.Groq(api_key=api_key)
        self.client = client

    def __call__(self, state: dict):

        user_inputs = state["user_constraints"]
        prompt = f"""
                    You are a strict JSON generator.
                    Your task:
                    - Validate and correct the user's optimization constraints.
                    - DO NOT add any explanation outside the JSON.
                    - DO NOT use markdown.
                    - DO NOT use backticks.
                    - DO NOT output anything except a JSON object.

                    User Constraints:
                    {user_inputs}

                    Return JSON ONLY in this EXACT format:
                    {{
                    "selected_objective": "<string>",
                    "roi_min": <float>,
                    "promo_share": <object>,
                    "horizon": <int>,
                    "explanation": "<string>"
                    }}
"""

        response = self.client.beta.chat.completions.parse(
            # model="llama-3.1-8b-instant",
            model = "gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format=ConstraintOutput
        )
        raw = response.choices[0].message.content.strip()
        print(f'[DEBUG] Raw LLM output for constraint selection:\n{raw}\n')

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            #  fallback
            print("[WARN] JSONDecodeError encountered, using fallback extraction.")
            parsed = extract_json(raw)

        # state["final_constraints"] = extract_json(raw)
        state["final_constraints"] = parsed

        print(f"[DEBUG] Parsed LLM Output - Final Constraints: {state['final_constraints']}")
        print("\n----------------------------------------------------------------------------")

        # parsed = ConstraintOutput.parse_raw(response.choices[0].message.content)
        # state["final_constraints"] = parsed.dict()
        return state



# ---------------------------------------------------------
# FINAL DATA NODE — Aggregates everything
# ---------------------------------------------------------

class FinalDataLayerNode:
    def __call__(self, state: dict):

        state["final_data_layer"] = {
            "filters": state["filters"],
            "budget": state["budget"],
            "objective": state["final_constraints"]["selected_objective"],
            "roi_min": state["final_constraints"]["roi_min"],
            "promo_share": state["final_constraints"]["promo_share"],
            "horizon": state["final_constraints"]["horizon"]
        }

        return state


# # ---------------------------------------------------------
# # BUILD GRAPH (ONE GRAPH WITH ALL NODES)
# # ---------------------------------------------------------

# def build_preparation_graph(api_key: str):
#     graph = StateGraph(dict)

#     graph.add_node("budget", BudgetSuggestionNode(api_key))
#     graph.add_node("constraints", ConstraintSelectionNode(api_key))
#     graph.add_node("final_data", FinalDataLayerNode())

#     graph.set_entry_point("budget")

#     graph.add_edge("budget", "constraints")
#     graph.add_edge("constraints", "final_data")
#     graph.add_edge("final_data", END)

#     return graph.compile()
