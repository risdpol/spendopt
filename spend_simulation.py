# spend_simulation.py
import pandas as pd
import numpy as np
from datetime import datetime
from config import OptimizerContext
from heavy_optimizer import rev_from_elasticity
# -------------------------------------------------------------------
# 1. LOCAL RE-OPTIMIZATION (NO MIP SOLVE)
# -------------------------------------------------------------------

def reoptimize_after_edit(
        df_modified: pd.DataFrame,
        model_store: dict,
        edited_index: int,
        new_row_indices: list[int] | None = None,
        ctx: object = None):

    out = df_modified.copy()
    EPS = ctx.EPS if ctx else 1e-6
    elasticity_map = model_store.get("elasticity", {})
    global_beta = model_store.get("global_beta", 1.0)
    baselines = model_store.get("baselines", {})
    avg_spend_sku_promo = model_store.get("avg_spend_sku_promo", {})
    avg_spend_sku_week  = model_store.get("avg_spend_sku_week", {})
    holiday_set = model_store.get("holiday_week_set", set())

    # --------------------------------------------------
    # Helper
    # --------------------------------------------------
    def compute_est_spend(sku, week, promo, old_spend, cap=True):
        promo_avg = avg_spend_sku_promo.get((sku, promo), None)
        week_avg  = avg_spend_sku_week.get((sku, week), None)

        if week_avg is None:
            sku_vals = [v for (k, v) in avg_spend_sku_promo.items() if k[0] == sku]
            if sku_vals:
                week_avg = float(np.mean(sku_vals))

        is_holiday = (week in holiday_set)
        w_promo, w_week = (0.5, 0.5) if is_holiday else (0.7, 0.3)

        if promo_avg is not None and week_avg is not None:
            est_spend = w_promo * promo_avg + w_week * week_avg
        elif promo_avg is not None:
            est_spend = promo_avg
        elif week_avg is not None:
            est_spend = week_avg
        else:
            est_spend = old_spend

        return est_spend

    # ----------------------------------------------------
    # STEP 1 — EDITED ROW
    # ----------------------------------------------------
    if edited_index not in out.index:
        raise ValueError("edited_index must exist in df_modified")

    e = out.loc[edited_index]
    sku, store, year, week, promo = (
        e["sku_id"], e["store_id"], int(e["year"]), int(e["week"]), e["promo_mechanics"]
    )
    old_spend = float(e["Planned_Spend"])

    new_spend = compute_est_spend(sku, week, promo, old_spend, cap=True)

    base_sp, base_rev = baselines.get((sku, store, year, week, promo), (None, None))
    if base_sp is None or base_rev is None:
        candidates = [v for (k, v) in baselines.items() if k[0] == sku]
        base_sp = np.median([v[0] for v in candidates])
        base_rev = np.median([v[1] for v in candidates])

    base_sp, base_rev = max(float(base_sp), EPS), max(float(base_rev), EPS)
    beta = elasticity_map.get((sku, store, promo), global_beta)

    new_metric = rev_from_elasticity(ctx, beta, new_spend, base_sp, base_rev)
    new_profit = new_metric - new_spend

    out.at[edited_index, "Planned_Spend"] = float(new_spend)
    out.at[edited_index, "Pred_Revenue"] = float(new_metric)
    out.at[edited_index, "Pred_Profit"] = float(new_profit)

    # ----------------------------------------------------
    # STEP 2 — NEW ROWS
    # ----------------------------------------------------
    if new_row_indices:
        for idx in new_row_indices:
            if idx not in out.index:
                continue

            r = out.loc[idx]
            sku_n, store_n = r["sku_id"], r["store_id"]
            year_n, week_n = int(r["year"]), int(r["week"])
            promo_n = r["promo_mechanics"]

            est_spend = compute_est_spend(sku_n, week_n, promo_n, old_spend=0.0, cap=False)

            bs, br = baselines.get((sku_n, store_n, year_n, week_n, promo_n), (None, None))
            if bs is None or br is None:
                candidates = [v for (k, v) in baselines.items() if k[0] == sku_n]
                bs = np.median([v[0] for v in candidates])
                br = np.median([v[1] for v in candidates])

            bs, br = max(float(bs), EPS), max(float(br), EPS)
            beta_n = elasticity_map.get((sku_n, store_n, promo_n), global_beta)

            metric_val = rev_from_elasticity(ctx, beta_n, est_spend, bs, br)
            profit_val = metric_val - est_spend

            out.at[idx, "Planned_Spend"] = float(est_spend)
            out.at[idx, "Pred_Revenue"] = float(metric_val)
            out.at[idx, "Pred_Profit"] = float(profit_val)

    return out


# -------------------------------------------------------------------
# 2. REBUILD (interactive editing)
# -------------------------------------------------------------------
def rebuild(plan_df, model_store, ctx):
    print("\nInitial Plan (first 10 rows):")
    print(plan_df.head(10))

    df1 = plan_df.copy()

    row_index = int(input("\nRow index to EDIT: "))
    if row_index not in df1.index:
        raise ValueError("Invalid row index.")

    new_week = int(input("Enter NEW Week: "))

    promo_options = sorted(df1["promo_mechanics"].unique())
    print("\nAvailable Promo Options:")
    for i, p in enumerate(promo_options, start=1):
        print(f"{i}. {p}")

    promo_idx = int(input("Enter Promo index: ")) - 1
    new_promo = promo_options[promo_idx]

    df1.at[row_index, "week"] = new_week
    df1.at[row_index, "promo_mechanics"] = new_promo

    # Add new rows
    add_n = int(input("\nHow many NEW rows to add after this edited row? (0,1,2...): "))
    new_row_indices = []

    if add_n > 0:
        base_row = df1.iloc[row_index].to_dict()
        new_rows_list = []

        for i in range(add_n):
            print(f"\n--- New Row #{i+1} ---")
            wk = int(input("Enter Week: "))

            print("Promo options:")
            for j, p in enumerate(promo_options, start=1):
                print(f"{j}. {p}")
            p_idx = int(input("Enter Promo index: ")) - 1
            pm = promo_options[p_idx]

            new_row = base_row.copy()
            new_row["week"] = wk
            new_row["promo_mechanics"] = pm
            new_row["Planned_Spend"] = 0.0
            new_row["Pred_Revenue"] = 0.0
            new_row["Pred_Profit"] = 0.0
            new_rows_list.append(new_row)

        upper = df1.iloc[:row_index+1]
        lower = df1.iloc[row_index+1:]
        df1 = pd.concat([upper, pd.DataFrame(new_rows_list), lower], ignore_index=True)
        new_row_indices = list(range(row_index+1, row_index+1+add_n))

    updated_plan = reoptimize_after_edit(
        df_modified=df1,
        model_store=model_store,
        edited_index=row_index,
        new_row_indices=new_row_indices,
        ctx=ctx
    )

    updated_plan = updated_plan.dropna(subset=["sku_id"]).reset_index(drop=True)
    # out_file = f"{datetime.now().strftime('%Y-%m-%d')}_revised_optimized_plan.csv"
    # updated_plan.to_csv(out_file, index=False)
    # print(f"[INFO] Saved")
    updated_summary = {
        # "Status": pulp.LpStatus[mdl.status],
        "Objective": ctx.objective,
        "Budget_Global": float(ctx.budget),
        "Budget_Used": float(sum(updated_plan['Planned_Spend'])),
        "Total_Revenue": float(sum(updated_plan['Pred_Revenue'])),
        "Total_Profit": float(sum(updated_plan['Pred_Profit'])),
        "Num_Choices": int(updated_plan.shape[0]),
    }
    print("[INFO] Updated Summary:", updated_summary)
    return updated_plan


# -------------------------------------------------------------------
# 3. LangGraph Node
# -------------------------------------------------------------------
# class SimulationNode:
#     """
#     This node accepts:
#         - plan (list or DataFrame)
#         - model_store
#     And returns:
#         - updated_plan
#     """

#     def __call__(self, state: dict):

#         print("\n[INFO] SIMULATION LAYER STARTED...")

#         plan = state["plan"]
#         model_store = state["model_store"]
#         ctx = state["ctx"]
#         plan_df = pd.DataFrame(plan)
#         updated = rebuild(plan_df, model_store, ctx)
#         state["updated_plan"] = updated.to_dict(orient="records")
#         print("[INFO] SIMULATION LAYER COMPLETE.")

#         return state

class SimulationNode:
    def __call__(self, state: dict):

        print("\n[INFO] SIMULATION LAYER STARTED...")

        model_store = state["model_store"]
        ctx = state["ctx"]

        # ---------- SAFE PLAN SELECTION ----------
        if "edited_plan" in state and state["edited_plan"] is not None:
            df_modified = pd.DataFrame(state["edited_plan"])
        else:
            df_modified = pd.DataFrame(state["plan"])

        # ---------- SAFE EDIT METADATA ----------
        edited_index = state.get("edited_index", 0)
        new_row_indices = state.get("new_row_indices", [])

        # ---------- RE-OPTIMIZE ----------
        updated_df = reoptimize_after_edit(
            df_modified=df_modified,
            model_store=model_store,
            edited_index=edited_index,
            new_row_indices=new_row_indices,
            ctx=ctx
        )

        updated_df = updated_df.dropna(subset=["sku_id"]).reset_index(drop=True)

        updated_summary = {
            "Objective": ctx.objective,
            "Budget_Global": float(ctx.budget),
            "Budget_Used": float(updated_df["Planned_Spend"].sum()),
            "Total_Revenue": float(updated_df["Pred_Revenue"].sum()),
            "Total_Profit": float(updated_df["Pred_Profit"].sum()),
            "Num_Choices": int(updated_df.shape[0]),
        }

        print("[INFO] SIMULATION LAYER COMPLETE.")

        return {
            "updated_plan": updated_df.to_dict(orient="records"),
            "updated_summary": updated_summary,
        }
