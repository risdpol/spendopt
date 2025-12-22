# heavy_optimizer.py
import numpy as np
import pandas as pd
import pulp
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from config import OptimizerContext

# ---------------------------
# Helper / small utilities
# ---------------------------

def ensure_numeric(series):
    """Coerce to numeric and return the series (with NaNs)."""
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# STEP 0b: INITIAL HISTORICAL FREQUENCY
# ============================================================
def step0b_initial_hist_freq(ctx: OptimizerContext):
    """
    Compute last historical year and historical promo frequency.
    Returns a dict: (sku_id, promo_mechanics) -> hist_weeks
    Preserves original prints.
    """
    df = ctx.df
    print("[INFO] Step 0b: Computing initial historical frequency (SKU, promo_mechanics)...")

    last_hist_year = int(df["year"].max())

    use_promoflag = "promoflag" in df.columns
    if use_promoflag:
        df_hist = df[(df["year"] == last_hist_year) & (df["promoflag"] == 1)]
    else:
        df_hist = df[(df["year"] == last_hist_year) & (df["promo_spends"] > 0)]

    hist_freq = (
        df_hist.groupby(["sku_id", "store_id", "promo_mechanics"])["week"]
               .nunique()
               .reset_index(name="hist_weeks")
    )

    max_weeks_dict_initial = {
        (row.sku_id, row.promo_mechanics): int(row.hist_weeks)
        for row in hist_freq.itertuples(index=False)
    }

    print(f"[INFO] Step 0b complete. Unique (sku,store,promo) pairs: {len(hist_freq)}")
    print("\n----------------------------------------------------------------------------")

    # store in context if desired
    ctx._hist_freq_initial = max_weeks_dict_initial
    ctx._last_hist_year_initial = last_hist_year

    return max_weeks_dict_initial


# ============================================================
# ELASTICITY FITTING (fit_elasticity_pair & step1_elasticities)
# ============================================================

def fit_elasticity_pair(ctx: OptimizerContext, sub: pd.DataFrame):
    """ log(Revenue+eps) ~ α + β*log(Spend+eps) + week dummies (Ridge)
        but adapted per OBJECTIVE (revenue/profit/npm/roi)
    """
    # SELECT TARGET COLUMN
    objective = ctx.objective
    if objective == "revenue":
        target_col = "total_revenue"
        apply_log = True
    elif objective == "profit":
        target_col = "net_profit"
        apply_log = False
    elif objective == "npm":
        target_col = "net_profit_margin"
        apply_log = False
    elif objective == "roi":
        target_col = "roi_pct"
        apply_log = False
    else:
        raise ValueError(f"Unsupported OBJECTIVE: {objective}")

    sub = sub.copy()

    # Clean target numeric
    sub[target_col] = ensure_numeric(sub[target_col])
    sub = sub.dropna(subset=[target_col])

    # Check minimum rows / variation
    if sub.shape[0] < ctx.MIN_ROWS_PER_PAIR:
        return None
    if sub["promo_spends"].nunique() < ctx.MIN_UNIQUE_SPEND:
        return None

    # Build X and y
    if apply_log:
        y = np.log(sub[target_col] + ctx.EPS)
    else:
        y = sub[target_col].astype(float)

    X = sub[["week", "promo_spends"]].copy()
    X["logSpend"] = np.log(sub["promo_spends"] + ctx.EPS)

    pre = ColumnTransformer(
        [("week", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["week"])],
        remainder="passthrough"  # keeps logSpend
    )
    model = Pipeline([("prep", pre), ("reg", Ridge(alpha=1.0, random_state=ctx.RANDOM_STATE))])

    try:
        model.fit(X[["week", "logSpend"]], y)
    except Exception as e:
        print(f"[WARN] Elasticity fit skipped due to regression error: {e}")
        return None

    beta = float(model.named_steps["reg"].coef_[-1])
    return {"model": model, "beta": beta}


def step1_elasticities(ctx: OptimizerContext):
    """
    Loop over ELASTICITY_KEY_COLS and fit elasticities.
    Populates ctx.beta_by_key and ctx.global_beta.
    Preserves print statements.
    """
    print("[INFO] Step 1: Fitting elasticities at (SKU, store, promo_mechanics)...")

    df = ctx.df
    elastic_rows = []
    models_by_pair = {}
    beta_by_key = {}

    for key_vals, sub in df.groupby(ctx.ELASTICITY_KEY_COLS):
        res = fit_elasticity_pair(ctx, sub)
        if res is None:
            continue

        key_dict = dict(zip(ctx.ELASTICITY_KEY_COLS, key_vals))
        models_by_pair[key_vals] = res["model"]
        beta_by_key[key_vals] = res["beta"]

        row_rec = key_dict.copy()
        row_rec["beta"] = res["beta"]
        elastic_rows.append(row_rec)

    elastic_df = pd.DataFrame(elastic_rows)
    if elastic_df.empty:
        raise RuntimeError("Not enough variation to estimate elasticities at (SKU,Promo_Type).")

    global_beta = float(elastic_df["beta"].median())

    # save to context
    ctx.beta_by_key = beta_by_key
    ctx.models_by_pair = models_by_pair
    ctx.global_beta = global_beta
    ctx.elastic_df = elastic_df

    print(f"[INFO] Step 1 complete. Elasticities estimated for {len(elastic_df)} (SKU,store,promo) combos.")
    print(f"[INFO] Global beta (median): {global_beta:.4f}")
    print("\n----------------------------------------------------------------------------")

    return ctx


# ============================================================
# STEP 2: HISTORICAL BASELINES + FUTURE HORIZON
# ============================================================
def step2_baselines_and_future(ctx: OptimizerContext, HORIZON_MONTHS: int = 3):
    """
    Build hist_base, pair_base, and create future baseline by crossing SKU–Promo with future weeks.
    Returns ctx with ctx.base_for_bp set and also returns the dataframe.
    """
    print("[INFO] Step 2: Building historical baselines and future horizon...")

    df = ctx.df

    # Convert months -> weeks (approx)
    WEEKS_PER_MONTH = 4.345
    FUTURE_WEEKS = int(round(HORIZON_MONTHS * WEEKS_PER_MONTH))
    USE_FUTURE_ONLY = True

    group_cols_hist = ctx.DECISION_KEY_COLS + ["promo_mechanics"]

    hist_base = (
        df.groupby(group_cols_hist)
          .agg(base_spend=("promo_spends", "median"),
               base_rev=("total_revenue", "median"))
          .reset_index()
    )

    if hist_base.empty:
        raise RuntimeError("No historical baselines found; check filters or data.")

    pair_cols = ["sku_id", "store_id", "promo_mechanics"]
    pair_base = (
        df.groupby(pair_cols)
          .agg(base_spend=("promo_spends", "median"),
               base_rev=("total_revenue", "median"))
          .reset_index()
    )

    if pair_base.empty:
        raise RuntimeError("No SKU–Promo baselines; cannot extend into future horizon.")

    last_year = int(df["year"].max())
    last_week = int(df.loc[df["year"] == last_year, "week"].max())

    def build_future_weeks(last_year: int, last_week: int, n_future_weeks: int):
        year, wk = last_year, last_week
        future = []
        for _ in range(n_future_weeks):
            wk += 1
            if wk > 52:
                wk = 1
                year += 1
            future.append((year, wk))
        return future

    future_weeks = build_future_weeks(last_year, last_week, FUTURE_WEEKS)

    print(f"[INFO] Last data year/week = {last_year}/{last_week}")
    print(f"[INFO] Building future horizon of {HORIZON_MONTHS} months "
          f"≈ {FUTURE_WEEKS} weeks -> {len(future_weeks)} (year,week) points")

    future_rows = []
    for r in pair_base.itertuples(index=False):
        sku_id = r.sku_id
        store_id = r.store_id
        promo  = r.promo_mechanics
        b_sp   = float(r.base_spend)
        b_rev  = float(r.base_rev)

        for (fy, fw) in future_weeks:
            future_rows.append({
                "sku_id":          sku_id,
                "store_id":        store_id,
                "year":            fy,
                "week":            fw,
                "promo_mechanics": promo,
                "base_spend":      b_sp,
                "base_rev":        b_rev,
            })

    future_base = pd.DataFrame(future_rows)

    if future_base.empty:
        raise RuntimeError("Future baseline table is empty; check horizon and pair_base.")

    print(f"[INFO] Future baseline rows created: {len(future_base)}")

    if USE_FUTURE_ONLY:
        base_for_bp = future_base.copy()
        print("[INFO] Using ONLY future weeks in the optimisation horizon.")
    else:
        base_for_bp = pd.concat([hist_base, future_base], ignore_index=True)
        print("[INFO] Using BOTH historical and future weeks in the optimisation horizon.")

    ctx.base_for_bp = base_for_bp

    # display(base_for_bp) in notebooks omitted
    print("[INFO] Step 2 complete.")
    print("\n----------------------------------------------------------------------------")

    return ctx


# ============================================================
# STEP 3: LEVEL BUILDER & REVENUE FUNCTION
# ============================================================
def build_levels(ctx: OptimizerContext, row_key, base_spend, nlevels=None):
    """
    Create spend levels for a given (SKU, Promo_Type).
    Ensures 0-spend level exists so optimizer can skip promo in a week.
    """
    if nlevels is None:
        nlevels = ctx.N_LEVELS

    sku_id = row_key["sku_id"]
    store_id = row_key["store_id"]
    promo  = row_key["promo_mechanics"]

    # --- Build quantile-based levels ---
    if ctx.LEVEL_RULE == "quantile":
        sub = ctx.df[(ctx.df["sku_id"] == sku_id) & (ctx.df["store_id"] == store_id) & (ctx.df["promo_mechanics"] == promo)]

        if sub.shape[0] >= nlevels:
            qs = np.linspace(0, 1, nlevels)
            steps = np.unique(np.quantile(sub["promo_spends"].values, qs)).tolist()
        else:
            b = max(base_spend, 1.0)
            base_range = [0.5 * b, 1.0 * b, 1.5 * b, 2.0 * b]
            steps = base_range[: max(1, nlevels - 1)]
    else:
        steps = [i * ctx.FIXED_STEP for i in range(1, nlevels)]

    if ctx.INCLUDE_ZERO_LEVEL:
        steps = [0.0] + steps

    steps = sorted({float(x) for x in steps if np.isfinite(x) and x >= 0.0})

    if len(steps) < nlevels:
        if len(steps) == 0:
            steps = [0.0]
        last = steps[-1]
        while len(steps) < nlevels:
            last += (steps[-1] * 0.2 + 1e-6)
            steps.append(last)

    return steps[:nlevels]


def rev_from_elasticity(ctx: OptimizerContext, beta, spend, base_sp, base_rev):
    """
    Applies the elasticity-based transformation depending on OBJECTIVE.
    Keeps behavior aligned with previous logic.
    """
    bs = max(base_sp, ctx.EPS)
    br = max(base_rev, ctx.EPS)
    s = max(float(spend), ctx.EPS)

    if ctx.objective in ["revenue", "profit"]:
        try:
            return float(br * ((s / bs) ** beta))
        except OverflowError:
            return 1e12
    else:
        try:
            delta = beta * np.log(s / bs)
            return float(br + delta)
        except Exception:
            return br


# ============================================================
# STEP 4: BUILD BREAKPOINTS (INCLUDING FUTURE WEEKS) and x variables
# ============================================================
def step4_build_breakpoints_and_x(ctx: OptimizerContext):
    """
    Build breakpoints from ctx.base_for_bp and elasticity map.
    Returns (mdl, x, idx_nonzero) to match prior signatures.
    """
    print("[INFO] Step 4: Building breakpoints and base MIP structure...")

    bp_rows = []
    base_for_bp = ctx.base_for_bp
    beta_by_key = ctx.beta_by_key or {}
    global_beta = ctx.global_beta

    for row in base_for_bp.itertuples(index=False):
        sku_id   = row.sku_id
        store_id = row.store_id
        year     = int(row.year)
        week     = int(row.week)
        promo    = row.promo_mechanics
        base_sp  = float(row.base_spend)
        base_rev = float(row.base_rev)

        row_key_elastic = {"sku_id": sku_id, "store_id": store_id ,"promo_mechanics": promo}
        key_tuple = tuple(row_key_elastic[col] for col in ctx.ELASTICITY_KEY_COLS)
        beta = beta_by_key.get(key_tuple, global_beta)

        level_key = {"sku_id": sku_id, "store_id": store_id ,"promo_mechanics": promo}
        levels = build_levels(ctx, level_key, base_spend=base_sp, nlevels=ctx.N_LEVELS)
        if not levels:
            continue

        for k_idx, s_k in enumerate(levels):
            r_k = rev_from_elasticity(ctx, beta, s_k, base_sp, base_rev)

            # ROI filter if needed
            if ctx.roi_min is not None:
                if s_k > 0.0:
                    roi_here = (r_k - s_k) / max(s_k, ctx.EPS)
                else:
                    roi_here = 0.0
                if roi_here < ctx.roi_min:
                    continue

            rec = {
                "sku_id":          sku_id,
                "store_id":        store_id,
                "year":            year,
                "week":            week,
                "Promo":           promo,
                "base_spend":      base_sp,
                "base_rev":        base_rev,
                "beta":            beta,
                "k":               int(k_idx),
                "s_k":             float(s_k),
                "r_k":             float(r_k),
            }
            bp_rows.append(rec)

    bp = pd.DataFrame(bp_rows)
    if bp.empty:
        print("[ERROR] No feasible breakpoints (all filtered). "
              "Relax ROI_MIN or level rules / check future horizon.")
        return None, None, None

    def make_itp(rec):
        parts = [str(rec[col]) for col in ctx.DECISION_KEY_COLS] + [str(rec["Promo"])]
        return "|".join(parts)

    bp["itp"] = bp.apply(make_itp, axis=1)
    triples = bp[ctx.DECISION_KEY_COLS + ["Promo", "itp"]].drop_duplicates()

    print(f"[INFO] Total breakpoints: {len(bp)}")
    print(f"[INFO] Total decision triples (SKU,year,week,Promo): {len(triples)}")

    # Create model
    mdl = pulp.LpProblem("Elasticity_PWL_MIP", pulp.LpMaximize)

    # Decision variables x[(itp,k)]
    x = {}
    for row in bp.itertuples(index=False):
        itp = row.itp
        k   = int(row.k)
        if (itp, k) not in x:
            x[(itp, k)] = pulp.LpVariable(name=f"x_{itp}_{k}", lowBound=0, upBound=1, cat="Binary")

    # Each itp -> exactly one chosen level
    for _, trow in triples.iterrows():
        itp = trow["itp"]
        ks = bp.loc[bp["itp"] == itp, "k"].unique().tolist()
        mdl += (pulp.lpSum(x[(itp, int(k))] for k in ks) == 1), f"one_level_{itp}"

    # Historical promo frequency (build local lookup)
    last_hist_year = int(ctx.df["year"].max())
    df_hist = ctx.df[ctx.df["year"] == last_hist_year].copy()
    if "promoflag" in df_hist.columns:
        df_hist = df_hist[df_hist["promoflag"] == 1]
    else:
        df_hist = df_hist[df_hist["promo_spends"] > 0]

    hist_freq = (
        df_hist.groupby(["sku_id", "store_id", "promo_mechanics"])["week"]
               .nunique()
               .reset_index(name="hist_weeks")
    )

    max_weeks_dict = {
        (row.sku_id, row.store_id, row.promo_mechanics): int(row.hist_weeks)
        for row in hist_freq.itertuples(index=False)
    }

    idx_nonzero = None
    for (sku, store, promo), grp in bp.groupby(["sku_id", "store_id", "Promo"]):
        hist_weeks = max_weeks_dict.get((sku, store, promo), 0)
        grp_future = grp[grp["year"] > last_hist_year]
        if grp_future.empty:
            continue

        idx_nonzero = grp_future[grp_future["s_k"] > 0]
        if idx_nonzero.empty:
            continue

        mdl += (
            pulp.lpSum(
                x[(row.itp, int(row.k))]
                for _, row in idx_nonzero.iterrows()
            ) <= hist_weeks,
            f"max_weeks_{sku}_{store}_{promo}"
        )

    # Save objects into context for later steps
    ctx.bp = bp
    ctx.triples = triples
    ctx.x = x

    print("[INFO] Step 4 complete.")
    print("\n----------------------------------------------------------------------------")
    return mdl, x, idx_nonzero


# ============================================================
# STEP 5: BUILD MIP MODEL
# ============================================================
def step5_build_mip(ctx: OptimizerContext, mdl, x):
    """
    Build full MIP using bp and triples stored in ctx.
    Returns mdl, y, lam.
    """
    bp = ctx.bp
    triples = ctx.triples

    print("[INFO] Step 5: Adding main MIP constraints and objective...")

    # Binary choice per (SKU,week,Promo)
    y = {
        row.itp: pulp.LpVariable(f"y_{row.itp}", lowBound=0, upBound=1, cat="Binary")
        for row in triples.itertuples(index=False)
    }

    # Convex weights per breakpoint
    lam = {
        (row.itp, int(row.k)): pulp.LpVariable(f"lam_{row.itp}_{int(row.k)}", lowBound=0, cat="Continuous")
        for row in bp.itertuples(index=False)
    }

    # Map for quick sums
    bp_idx_by_itp = defaultdict(list)
    for r in bp.itertuples(index=False):
        bp_idx_by_itp[r.itp].append(int(r.k))

    # (A) Convexity: sum_k lambda = y_itp
    for itp, ks in bp_idx_by_itp.items():
        mdl += pulp.lpSum(lam[(itp, k)] for k in ks) == y[itp], f"convexity_{itp}"

    # (B) At most one promo per (SKU,week)
    for key_vals, grp in triples.groupby(ctx.DECISION_KEY_COLS):
        key_vals_t = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        mdl += pulp.lpSum(y[itp] for itp in grp["itp"]) <= 1, \
               "one_promo_" + "_".join(str(v) for v in key_vals_t)

    # (C) Must-run / must-not-run at (SKU,week)
    def match_key(mask_df, key_tuple):
        mask = pd.Series(True, index=mask_df.index)
        for col, val in zip(ctx.DECISION_KEY_COLS, key_tuple):
            mask &= (mask_df[col] == val)
        return mask

    for key in ctx.MUST_RUN:
        mask = match_key(triples, key)
        its = triples.loc[mask, "itp"].tolist()
        if its:
            mdl += pulp.lpSum(y[it] for it in its) == 1, "must_run_" + "_".join(str(v) for v in key)

    for key in ctx.MUST_NOT_RUN:
        mask = match_key(triples, key)
        its = triples.loc[mask, "itp"].tolist()
        if its:
            mdl += pulp.lpSum(y[it] for it in its) == 0, "must_not_run_" + "_".join(str(v) for v in key)

    # (D) Global budget
    spend_terms = [r.s_k * lam[(r.itp, int(r.k))] for r in bp.itertuples(index=False)]
    mdl += pulp.lpSum(spend_terms) <= ctx.budget, "Budget_Global"

    # ========== Promo Share Constraint ==========
    if ctx.promo_share is not None:
        total_spend_expr = pulp.lpSum(spend_terms)
        for promo_name, share in ctx.promo_share.items():
            promo_spend_terms = [
                r.s_k * lam[(r.itp, int(r.k))]
                for r in bp.itertuples(index=False)
                if r.Promo == promo_name
            ]

            if promo_spend_terms:
                mdl += (
                    pulp.lpSum(promo_spend_terms)
                    == float(share) * total_spend_expr,
                    f"promo_share_{promo_name}"
                )

    # (E) Per-SKU cap (max # active weeks)
    if ctx.PER_SKU_MAX_WEEKS is not None:
        if isinstance(ctx.PER_SKU_MAX_WEEKS, int):
            per_sku_limit = defaultdict(lambda: ctx.PER_SKU_MAX_WEEKS)
        elif isinstance(ctx.PER_SKU_MAX_WEEKS, dict):
            per_sku_limit = defaultdict(lambda: 10**9, ctx.PER_SKU_MAX_WEEKS)
        else:
            raise ValueError("PER_SKU_MAX_WEEKS must be int, dict, or None.")

        # Group by sku_id
        for sku, grp in triples.groupby("sku_id"):
            mdl += pulp.lpSum(y[itp] for itp in grp["itp"]) <= per_sku_limit[sku], f"cap_sku_{sku}"

    # (F) Per-Promo cap (global)
    if ctx.PER_PROMO_CAP is not None:
        promo_limit = defaultdict(lambda: 10**9, ctx.PER_PROMO_CAP)
        for promo, grp in triples.groupby("Promo"):
            mdl += pulp.lpSum(y[itp] for itp in grp["itp"]) <= promo_limit[promo], f"cap_promo_{promo}"

    # (G) weekly budgets (optional)
    if ctx.WEEKLY_BUDGETS is not None:
        for week, cap in ctx.WEEKLY_BUDGETS.items():
            week = int(week)
            itps = triples[triples["week"] == week]["itp"].tolist()
            if not itps:
                continue
            weekly_terms = [
                r.s_k * lam[(r.itp, int(r.k))]
                for r in bp.itertuples(index=False) if r.itp in itps
            ]
            mdl += pulp.lpSum(weekly_terms) <= float(cap), f"budget_week_{week}"

    # ============================================================
    # (H) Historical promo frequency cap for FUTURE WEEKS ONLY
    # ============================================================
    last_hist_year = int(ctx.df["year"].max())

    df_hist = ctx.df[ctx.df["year"] == last_hist_year].copy()
    if "promoflag" in df_hist.columns:
        df_hist = df_hist[df_hist["promoflag"] == 1]
    else:
        df_hist = df_hist[df_hist["promo_spends"] > 0]

    hist_freq = (
        df_hist.groupby(["sku_id", "store_id", "promo_mechanics"])["week"]
               .nunique()
               .reset_index(name="hist_weeks")
    )

    max_weeks_dict = {
        (r.sku_id, r.store_id, r.promo_mechanics): int(r.hist_weeks)
        for r in hist_freq.itertuples(index=False)
    }

    print("[INFO] Historical promo frequency constraints (y-based) loaded.")

    for (sku, store, promo), grp in triples.groupby(["sku_id", "store_id", "Promo"]):
        hist_weeks = max_weeks_dict.get((sku, store, promo), 0)
        future_itps = grp[grp["year"] > last_hist_year]["itp"].tolist()
        if not future_itps:
            continue
        mdl += (
            pulp.lpSum(y[itp] for itp in future_itps) <= hist_weeks,
            f"hist_freq_cap_{sku}_{store}_{promo}"
        )

    # Objective: dynamic based on objective
    def profit_coef(r):
        if ctx.objective == "revenue":
            return r.r_k
        if ctx.objective == "profit":
            return r.r_k - r.s_k
        if ctx.objective == "npm":
            return r.r_k
        if ctx.objective == "roi":
            return r.r_k
        raise ValueError(f"Unsupported OBJECTIVE: {ctx.objective}")

    obj_terms = [profit_coef(r) * lam[(r.itp, int(r.k))] for r in bp.itertuples(index=False)]
    mdl += pulp.lpSum(obj_terms), "Objective"

    print("[INFO] Step 5 complete. MIP model fully built.")
    print("\n----------------------------------------------------------------------------")

    # Save y/lam to context for potential inspection
    ctx.y = y
    ctx.lam = lam

    return mdl, y, lam


# ============================================================
# STEP 6: SOLVE
# ============================================================
def step6_solve(mdl):
    print("[INFO] Step 6: Solving MIP model...")
    solver = pulp.PULP_CBC_CMD(msg=False)
    _ = mdl.solve(solver)
    print("[INFO] Solver status:", pulp.LpStatus[mdl.status])
    print("[INFO] Step 6 complete.")
    print("\n----------------------------------------------------------------------------")
    return pulp.LpStatus[mdl.status]


# ============================================================
# STEP 7: EXTRACT PLAN
# ============================================================
def step7_extract_plan(ctx: OptimizerContext, mdl, y, lam):
    print("[INFO] Step 7: Extracting optimal plan...")

    bp = ctx.bp
    triples = ctx.triples

    plan_rows = []
    for key_vals, grp in triples.groupby(ctx.DECISION_KEY_COLS):
        chosen_itp, best = None, -1
        for itp in grp["itp"]:
            val = y[itp].value()
            if val is not None and val > best:
                best, chosen_itp = val, itp
        if chosen_itp is None or best < 0.5:
            continue

        sub = bp[bp["itp"] == chosen_itp]
        spend_val = sum(r.s_k * (lam[(r.itp, int(r.k))].value() or 0.0) for r in sub.itertuples(index=False))
        rev_val = sum(r.r_k * (lam[(r.itp, int(r.k))].value() or 0.0) for r in sub.itertuples(index=False))
        promo = sub["Promo"].iloc[0]

        row_dict = {}
        if isinstance(key_vals, tuple):
            for col, v in zip(ctx.DECISION_KEY_COLS, key_vals):
                row_dict[col] = v
        else:
            row_dict[ctx.DECISION_KEY_COLS[0]] = key_vals

        row_dict.update({
            "promo_mechanics": promo,
            "Planned_Spend": float(spend_val),
            "Pred_Revenue": float(rev_val),
        })

        sku_id = row_dict["sku_id"]
        row_dict["Pred_Profit"] = float(
            (rev_val - spend_val)
            if ctx.MARGIN_BY_SKU is None
            else (float(ctx.MARGIN_BY_SKU.get(sku_id, 1.0)) * rev_val - spend_val)
        )

        plan_rows.append(row_dict)

    plan = pd.DataFrame(plan_rows).sort_values(ctx.DECISION_KEY_COLS)
    total_spend = plan["Planned_Spend"].sum() if not plan.empty else 0.0
    total_rev = plan["Pred_Revenue"].sum() if not plan.empty else 0.0
    total_profit = plan["Pred_Profit"].sum() if not plan.empty else 0.0

    summary = {
        "Status": pulp.LpStatus[mdl.status],
        "Objective": ctx.objective,
        "Budget_Global": float(ctx.budget),
        "Budget_Used": float(total_spend),
        "Total_Revenue": float(total_rev),
        "Total_Profit": float(total_profit),
        "Num_Choices": int(plan.shape[0]) if not plan.empty else 0,
    }

    print("[INFO] Summary:", summary)
    print("[INFO] Step 7 complete.")
    print("\n----------------------------------------------------------------------------")

    return plan, summary


# ============================================================
# BUILD MODEL STORE
# ============================================================
import pickle

def build_model_store(ctx: OptimizerContext):
    """
    Build a model_store dict from the heavy optimization artifacts held in ctx.
    """
    df = ctx.df
    base_for_bp = ctx.base_for_bp
    beta_by_key = ctx.beta_by_key
    global_beta = ctx.global_beta

    print("[INFO] Building model_store for re-optimization...")

    # 1) Elasticity map
    elasticity_map = {k: float(v) for k, v in (beta_by_key or {}).items()}

    # 2) Baselines for each (sku, store, year, week, promo)
    baselines = {}
    for r in base_for_bp.itertuples(index=False):
        baselines[(r.sku_id, r.store_id, int(r.year), int(r.week), r.promo_mechanics)] = (float(r.base_spend), float(r.base_rev))

    # 3) Promo-wise average spend per SKU
    df_nonzero = df[df["promo_spends"] > 0].copy()
    grp_promo = (
        df_nonzero
        .groupby(["sku_id", "promo_mechanics"])["promo_spends"]
        .mean()
        .reset_index(name="avg_spend")
    )
    avg_spend_sku_promo = {
        (r.sku_id, r.promo_mechanics): float(r.avg_spend)
        for r in grp_promo.itertuples(index=False)
    }

    # 4) Week-wise average spend per SKU
    grp_week = (
        df_nonzero
        .groupby(["sku_id", "week"])["promo_spends"]
        .mean()
        .reset_index(name="avg_spend")
    )
    avg_spend_sku_week = {
        (r.sku_id, int(r.week)): float(r.avg_spend)
        for r in grp_week.itertuples(index=False)
    }

    holiday_week_set = set()
    if "holiday_flag" in df.columns:
        holiday_week_set = set(df.loc[df["holiday_flag"] == 1, "week"].astype(int).unique())

    model_store = {
        "elasticity": elasticity_map,
        "global_beta": float(global_beta) if global_beta is not None else None,
        "baselines": baselines,
        "avg_spend_sku_promo": avg_spend_sku_promo,
        "avg_spend_sku_week": avg_spend_sku_week,
        "holiday_week_set": holiday_week_set
    }

    # Save model store to a pickled file path if ctx provides it
    if getattr(ctx, "MODEL_STORE_PATH", None):
        with open(ctx.MODEL_STORE_PATH, "wb") as f:
            pickle.dump(model_store, f)
        print("[INFO] Step 8 complete. Model store saved to:", ctx.MODEL_STORE_PATH)

    print("[INFO] model_store built:")
    print(f"  Elasticity entries:       {len(elasticity_map)}")
    print(f"  Baseline entries:         {len(baselines)}")
    print(f"  avg_spend_sku_promo keys: {len(avg_spend_sku_promo)}")
    print(f"  avg_spend_sku_week keys:  {len(avg_spend_sku_week)}")
    print("[INFO] Step 8 complete.")
    print("\n----------------------------------------------------------------------------")

    return model_store
