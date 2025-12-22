import streamlit as st
import pandas as pd

from streamlit_backend import run_pipeline, build_simulation_graph

st.set_page_config(layout="wide")
st.title("LLM Powered Spend Optimization")

# ================= SESSION INIT ================= #
if "step" not in st.session_state:
    st.session_state.step = 0

if "inputs" not in st.session_state:
    st.session_state.inputs = {}

if "result" not in st.session_state:
    st.session_state.result = None

# =================================================
# STEP 0: FILE UPLOAD
# =================================================
st.header("1. Upload Input Data")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file and st.session_state.step == 0:
    with open("temp.xlsx", "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.inputs["data_path"] = "temp.xlsx"
    st.session_state.step = 1
    st.rerun()

# =================================================
# STEP 1: CONSTRAINTS
# =================================================
if st.session_state.step >= 1:
    st.header("2. Budget & Constraints")

    objective = st.radio(
        "Objective",
        ["revenue", "profit", "roi", "npm"],
        index=0
    )

    roi_min = st.number_input(
        "ROI Min (e.g. 0.5 = 50%)",
        min_value=0.0,
        value=0.5,
        step=0.05
    )

    st.subheader("Promo Share (auto-normalized)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        discount = st.number_input("Discount",min_value=0.0,value=1.0)
    with c2:
        display = st.number_input("Display",min_value=0.0,value=1.0)
    with c3:
        bogo = st.number_input("BOGO",min_value=0.0,value=1.0)
    with c4:
        feature = st.number_input("Feature",min_value=0.0,value=1.0)
    
    PROMO_SHARE = {
        "discount": discount,
        "display": display,
        "bogo": bogo,
        "feature": feature,
    }

    total_share = sum(PROMO_SHARE.values())

    st.write(f"Original total share = {total_share}")

    # ---------------- NORMALIZATION (EXACT CLI LOGIC) ---------------- #
    if abs(total_share - 1) > 1e-6:
        st.info("Auto-normalizing promo shares...")

        for k in PROMO_SHARE:
            PROMO_SHARE[k] = PROMO_SHARE[k] / total_share

        st.write(f"New normalized total = {sum(PROMO_SHARE.values())}")
    else:
        st.info("Shares already sum to 1. No normalization needed.")

    st.subheader("Final Promo Share (Normalized)")

    promo_df = pd.DataFrame(
        PROMO_SHARE.items(),
        columns=["Promo Type", "Share"]
    )

    promo_df["Share"] = promo_df["Share"].round(3)

    st.dataframe(
        promo_df,
        use_container_width=True,
        column_config={
            "Share": st.column_config.ProgressColumn(
                "Share",
                min_value=0.0,
                max_value=1.0,
                format="%.2f"
            )
        }
    )

    # st.json(PROMO_SHARE)

    horizon = st.selectbox(
        "Optimization Horizon (months)",
        [3, 6, 9, 12],
        index=0
    )

    if st.button("Run Optimization", type="primary"):
        st.session_state.inputs.update({
            "objective": objective,
            "roi_min": roi_min,
            "promo_share": PROMO_SHARE,
            "horizon": horizon,
        })
        st.session_state.step = 2
        st.rerun()

# =================================================
# STEP 2: RUN OPTIMIZATION
# =================================================
if st.session_state.step == 2:
    st.info("Running optimization pipeline...")

    try:
        result = run_pipeline(**st.session_state.inputs)
        st.session_state.result = result
        st.session_state.step = 3
        st.success("✅ Optimization completed")
        st.rerun()

    except Exception as e:
        st.error("❌ Pipeline failed")
        st.exception(e)

# =================================================
# STEP 3: RESULTS
# =================================================
if st.session_state.step == 3 and st.session_state.result:
    result = st.session_state.result

    st.header("3. Optimization Results")

    # st.subheader("Budget Details")
    # st.json(result["budget_details"])

    st.subheader("Budget Overview")

    budget = result["budget_details"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Recommended Budget", f"{budget['recommended_budget']:.2f}")
    c2.metric("Lower Limit", f"{budget['lower_limit']:.2f}")
    c3.metric("Upper Limit", f"{budget['upper_limit']:.2f}")

    st.caption(budget.get("explanation", ""))


    # st.subheader("Optimization Summary")
    # st.json(result["summary"])

    st.subheader("Optimization Summary")

    summary = result["summary"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Budget Used",
        f"{summary.get('Budget_Used', 0):,.2f}"
    )

    c2.metric(
        "Total Revenue",
        f"{summary.get('Total_Revenue', 0):,.2f}"
    )

    c3.metric(
        "Total Profit",
        f"{summary.get('Total_Profit', 0):,.2f}"
    )

    st.divider()

    c4, c5 = st.columns(2)

    c4.metric(
        "Optimization Objective",
        summary.get("Objective", "NA")
    )

    c5.metric(
        "Number of Promo Decisions",
        summary.get("Num_Choices", 0)
    )


    plan_df = pd.DataFrame(result["plan"])

    st.subheader("Optimized Plan (Preview)")
    st.dataframe(plan_df.head(10), use_container_width=True)

    # =================================================
    # USER DECISION (CLI PARITY)
    # =================================================
    st.divider()
    st.subheader("Do you want to add / edit anything?")

    user_choice = st.radio(
        "Choose:",
        ["No, proceed with this plan", "Yes, open Spend Simulator"],
        index=0
    )

    # =================================================
    # STEP 4: SPEND SIMULATOR (GUIDED)
    # =================================================
    if user_choice.startswith("Yes"):
        st.header("4. Spend Simulator")

        # ---------- Step 1: Row index ----------
        edited_index = st.number_input(
            "Row index to EDIT",
            min_value=0,
            max_value=len(plan_df) - 1,
            value=0
        )

        # ---------- Step 2: New week ----------
        new_week = st.number_input(
            "Enter NEW Week",
            min_value=1,
            max_value=52,
            value=int(plan_df.loc[edited_index, "week"])
        )

        # ---------- Step 3: Promo ----------
        promo_options = sorted(plan_df["promo_mechanics"].unique())

        promo_choice = st.radio(
            "Available Promo Options",
            promo_options,
            index=promo_options.index(
                plan_df.loc[edited_index, "promo_mechanics"]
            )
        )

        # ---------- Step 4: Add rows ----------
        add_n = st.number_input(
            "How many NEW rows to add after this edited row?",
            min_value=0,
            max_value=5,
            value=0
        )

        # ---------- Step 5: Define new rows ----------
        new_rows = []
        if add_n > 0:
            st.subheader("Define New Rows")
            for i in range(add_n):
                st.markdown(f"**New Row #{i+1}**")

                wk = st.number_input(
                    f"Week (Row {i+1})",
                    min_value=1,
                    max_value=52,
                    key=f"new_week_{i}"
                )

                pm = st.radio(
                    f"Promo (Row {i+1})",
                    promo_options,
                    key=f"new_promo_{i}"
                )

                new_rows.append({
                    "week": wk,
                    "promo_mechanics": pm
                })

        # ---------- Build edited plan ----------
        edited_df = plan_df.copy()
        edited_df.at[edited_index, "week"] = new_week
        edited_df.at[edited_index, "promo_mechanics"] = promo_choice

        new_row_indices = []

        if new_rows:
            base_row = edited_df.loc[edited_index].to_dict()
            insert_rows = []

            for r in new_rows:
                nr = base_row.copy()
                nr["week"] = r["week"]
                nr["promo_mechanics"] = r["promo_mechanics"]
                nr["Planned_Spend"] = 0.0
                nr["Pred_Revenue"] = 0.0
                nr["Pred_Profit"] = 0.0
                insert_rows.append(nr)

            upper = edited_df.iloc[:edited_index+1]
            lower = edited_df.iloc[edited_index+1:]
            edited_df = pd.concat(
                [upper, pd.DataFrame(insert_rows), lower],
                ignore_index=True
            )

            new_row_indices = list(
                range(edited_index+1, edited_index+1+add_n)
            )

        # ---------- Run simulation ----------
        if st.button("Run Spend Simulation", type="primary"):
            sim_graph = build_simulation_graph()

            sim_out = sim_graph.invoke({
                "plan": result["plan"],
                "edited_plan": edited_df.to_dict(orient="records"),
                "edited_index": edited_index,
                "new_row_indices": new_row_indices,
                "model_store": result["model_store"],
                "ctx": result["ctx"],
            })

            # st.subheader("Updated Summary")
            # st.json(sim_out["updated_summary"])
            st.subheader("Updated Summary (After Simulation)")

            updated = sim_out["updated_summary"]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Objective",
                updated.get("Objective", "NA")
            )

            c2.metric(
                "Budget Used",
                f"{updated.get('Budget_Used', 0):,.2f}"
            )

            c3.metric(
                "Total Revenue",
                f"{updated.get('Total_Revenue', 0):,.2f}"
            )

            c4.metric(
                "Total Profit",
                f"{updated.get('Total_Profit', 0):,.2f}"
            )

            st.divider()

            c5, c6 = st.columns(2)

            c5.metric(
                "Global Budget",
                f"{updated.get('Budget_Global', 0):,.2f}"
            )

            c6.metric(
                "Number of Promo Decisions",
                updated.get("Num_Choices", 0)
            )

            st.subheader("Updated Plan")
            st.dataframe(
                pd.DataFrame(sim_out["updated_plan"]),
                use_container_width=True
            )

    # =================================================
    # RESET
    # =================================================
    st.divider()
    if st.button("Start New Run"):
        st.session_state.clear()
        st.rerun()
