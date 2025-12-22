# model_building.py
import pandas as pd
import numpy as np
from pydantic import BaseModel

# ============================================================
# MODEL BUILDING NODE (LangGraph)
# ============================================================

# from config import OptimizerContext

# model_building.py
from heavy_optimizer import (
    OptimizerContext,
    step0b_initial_hist_freq,
    step1_elasticities,
    step2_baselines_and_future,
    step4_build_breakpoints_and_x,
    step5_build_mip,
    step6_solve,
    step7_extract_plan,
    build_model_store
)

class ModelOutput(BaseModel):
    plan: list        # list of dicts (plan rows)
    summary: dict
    model_store: dict

class ModelBuildingNode:

    def __call__(self, state: dict):

        print("\n[INFO] MODEL BUILDING NODE STARTED...")

        df = state["filtered_df"]
        budget = state["budget"]
        objective = state["objective"]
        roi_min = state["roi_min"]
        promo_share = state["promo_share"]
        horizon_months = state["HORIZON_MONTHS"]

        # Create context here only
        ctx = OptimizerContext(
            df=df,
            budget=budget,
            objective=objective,
            roi_min=roi_min,
            promo_share=promo_share
        )

        # Call optimizer pipeline
        step0b_initial_hist_freq(ctx)
        step1_elasticities(ctx)
        step2_baselines_and_future(ctx, horizon_months)

        mdl, x, idx_nz = step4_build_breakpoints_and_x(ctx)
        mdl, y, lam = step5_build_mip(ctx, mdl, x)

        status = step6_solve(mdl)
        plan, summary = step7_extract_plan(ctx, mdl, y, lam)

        # Build model_store
        model_store = build_model_store(ctx)

        return {
            "plan": plan.to_dict(orient="records"),
            "summary": summary,
            "model_store": model_store,
            "ctx": ctx
        }
