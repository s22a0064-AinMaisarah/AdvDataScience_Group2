import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Pasar Mini Prescriptive Analysis",
    layout="wide",
    page_icon="🧠",
)

# -------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------
st.sidebar.title("⚙️ Controls")

st.sidebar.markdown("Configure prescriptive analysis settings:")

w_access = st.sidebar.slider(
    "Weight: Access (inverse outlet density)", 0.0, 1.0, 0.4, 0.05
)
w_price = st.sidebar.slider(
    "Weight: Price risk (avg basket price)", 0.0, 1.0, 0.3, 0.05
)
w_demand = st.sidebar.slider(
    "Weight: Demand (transaction volume)", 0.0, 1.0, 0.3, 0.05
)

top_n_tier1 = st.sidebar.number_input("Number of Tier 1 states", 1, 8, 5)
top_n_tier2 = st.sidebar.number_input("Number of Tier 2 states", 1, 8, 5)
N_new_outlets = st.sidebar.number_input(
    "Scenario: extra outlets per Tier 1 state", 0, 100, 20
)

# -------------------------------------------------------
# Title and objectives
# -------------------------------------------------------
st.title("🧠 Prescriptive Analysis – Pasar Mini")

st.write(
    "This page provides data-driven recommendations for **Pasar Mini** "
    "outlet expansion and logistics support using the December 2025 "
    "PriceCatcher dataset."
)

st.subheader("🎯 Objectives of Prescriptive Analysis")
st.markdown(
    """
    - **Objective 1:** Generate data-driven, actionable recommendations for Pasar Mini
      outlet expansion and logistics support across Malaysian states.
    - **Objective 2:** Optimise the allocation of limited outlets and support resources
      using a Priority Index based on access, price risk, and demand.
    - **Objective 3:** Evaluate alternative policy scenarios (e.g., outlet expansion in
      Tier 1 states) and identify strategies that maximise improvements in food access
      and affordability.
    """
)

# -------------------------------------------------------
# 1. Load and join data
# -------------------------------------------------------
@st.cache_data
def load_data():
    URL_PRICECATCHER = "https://storage.data.gov.my/pricecatcher/pricecatcher_2025-12.parquet"
    URL_PREMISE = "https://storage.data.gov.my/pricecatcher/lookup_premise.parquet"
    URL_ITEM = "https://storage.data.gov.my/pricecatcher/lookup_item.parquet"

    price_df = pd.read_parquet(URL_PRICECATCHER)
    premise_df = pd.read_parquet(URL_PREMISE)
    item_df = pd.read_parquet(URL_ITEM)

    df = (
        price_df.merge(premise_df, on="premise_code", how="left")
                .merge(item_df, on="item_code", how="left")
    )
    df["date"] = pd.to_datetime(df["date"])
    return df

with st.spinner("Loading PriceCatcher and lookup tables..."):
    df = load_data()

pm_df = df[df["premise_type"] == "Pasar Mini"].copy()

# -------------------------------------------------------
# 2. Basket selection
# -------------------------------------------------------
st.subheader("🧺 Staple Basket Selection")

all_items = sorted(pm_df["item"].dropna().str.lower().unique())
default_basket = [
    "minyak masak tulen cap buruh",
    "beras cap rambutan (sst5%)",
    "bawang besar kuning/holland",
    "timun",
]
default_basket = [i for i in default_basket if i in all_items]

basket_items = st.multiselect(
    "Select staple items to represent the basic food basket:",
    options=all_items,
    default=default_basket,
    help="These items are used to compute average basket prices by state.",
)

if len(basket_items) == 0:
    st.warning("Please select at least one staple item to continue.")
    st.stop()

pm_df["item_lower"] = pm_df["item"].str.lower()
basket_df = pm_df[pm_df["item_lower"].isin(basket_items)].copy()

# -------------------------------------------------------
# 3. State metrics & Priority Index
# -------------------------------------------------------
state_premises = pm_df.groupby("state")["premise"].nunique().rename("premise_count")
state_txn = pm_df.groupby("state")["price"].count().rename("txn_count")
state_basket_price = basket_df.groupby("state")["price"].mean().rename("avg_basket_price")

state_stats = pd.concat(
    [state_premises, state_txn, state_basket_price], axis=1
).dropna()

state_stats = state_stats.copy()
state_stats = state_stats.fillna(state_stats.median(numeric_only=True))

scaler = MinMaxScaler()
state_stats["premise_norm"] = scaler.fit_transform(state_stats[["premise_count"]])
state_stats["price_norm"]   = scaler.fit_transform(state_stats[["avg_basket_price"]])
state_stats["txn_norm"]     = scaler.fit_transform(state_stats[["txn_count"]])

state_stats["access_score"] = 1 - state_stats["premise_norm"]
state_stats["price_score"]  = state_stats["price_norm"]
state_stats["demand_score"] = state_stats["txn_norm"]

w_sum = w_access + w_price + w_demand
w_access_n, w_price_n, w_demand_n = (
    w_access / w_sum,
    w_price / w_sum,
    w_demand / w_sum,
) if w_sum > 0 else (0, 0, 0)

state_stats["priority_index"] = (
    w_access_n * state_stats["access_score"]
    + w_price_n * state_stats["price_score"]
    + w_demand_n * state_stats["demand_score"]
)

state_stats_sorted = state_stats.sort_values("priority_index", ascending=False)
state_stats_sorted["state"] = state_stats_sorted.index

# quick metrics
total_pm = int(pm_df["premise"].nunique())
tier1_states_preview = ", ".join(state_stats_sorted.head(top_n_tier1).index)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Pasar Mini outlets", f"{total_pm:,}")
col_b.metric("Number of states", f"{state_stats_sorted.shape[0]}")
col_c.metric("Top priority states", tier1_states_preview)

# -------------------------------------------------------
# Tabs for visuals
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Priority Overview", "🧩 Component Scores", "🧪 Scenario"])

# ---------- Tab 1: Priority Overview ----------
with tab1:
    st.subheader("State Priority Index")

    chart_data = state_stats_sorted[["state", "priority_index"]]
    chart = (
        alt.Chart(chart_data)
        .mark_bar(color="#FF7F0E")
        .encode(
            x=alt.X("state", sort="-y", title="State"),
            y=alt.Y("priority_index", title="Priority Index"),
            tooltip=["state", "priority_index"],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "States with higher Priority Index have fewer Pasar Mini outlets, higher "
        "basket prices, and/or stronger demand, and are therefore higher priority "
        "for intervention."
    )

    st.write("Detailed table:")
    st.dataframe(
        state_stats_sorted[
            ["premise_count","txn_count","avg_basket_price",
             "access_score","price_score","demand_score","priority_index"]
        ]
    )

# ---------- Tab 2: Component scores ----------
with tab2:
    st.subheader("Component Scores by State")

    comp_df = state_stats_sorted.melt(
        id_vars="state",
        value_vars=["access_score","price_score","demand_score"],
        var_name="component",
        value_name="score",
    )

    comp_chart = (
        alt.Chart(comp_df)
        .mark_bar()
        .encode(
            x=alt.X("state", sort="-y", title="State"),
            y=alt.Y("score", title="Normalised score"),
            color=alt.Color("component", title="Component"),
            tooltip=["state","component","score"],
        )
        .properties(height=400)
    )
    st.altair_chart(comp_chart, use_container_width=True)

# ---------- Tab 3: Scenario analysis ----------
with tab3:
    st.subheader("Outlet Expansion Scenario for Tiered States")

    # tiering
    state_stats_sorted["rank"] = state_stats_sorted["priority_index"].rank(
        ascending=False, method="first"
    )

    def assign_tier(rank):
        if rank <= top_n_tier1:
            return "Tier 1"
        elif rank <= top_n_tier1 + top_n_tier2:
            return "Tier 2"
        else:
            return "Tier 3"

    state_stats_sorted["tier"] = state_stats_sorted["rank"].apply(assign_tier)

    st.write("Current tiers:")
    st.dataframe(
        state_stats_sorted[
            ["premise_count","avg_basket_price","txn_count",
             "priority_index","tier"]
        ]
    )

    # scenario: add outlets to Tier 1
    scenario = state_stats_sorted.copy()
    tier1_states = scenario[scenario["tier"] == "Tier 1"]["state"]

    scenario.loc[scenario["state"].isin(tier1_states), "premise_count"] += N_new_outlets

    scenario["premise_norm"] = scaler.fit_transform(scenario[["premise_count"]])
    scenario["access_score"] = 1 - scenario["premise_norm"]
    scenario["priority_index_new"] = (
        w_access_n * scenario["access_score"]
        + w_price_n * scenario["price_score"]
        + w_demand_n * scenario["demand_score"]
    )
    scenario["delta_priority"] = scenario["priority_index_new"] - scenario["priority_index"]

    st.markdown("**Impact table (negative delta = improvement):**")
    st.dataframe(
        scenario[["state","tier","priority_index","priority_index_new","delta_priority"]]
        .sort_values("delta_priority")
        .reset_index(drop=True)
    )

    scenario_melt = scenario.melt(
        id_vars=["state"],
        value_vars=["priority_index","priority_index_new"],
        var_name="scenario",
        value_name="value",
    )

    scenario_chart = (
        alt.Chart(scenario_melt)
        .mark_bar()
        .encode(
            x=alt.X("state", sort="-y", title="State"),
            y=alt.Y("value", title="Priority Index"),
            color=alt.Color("scenario", title="Scenario"),
            tooltip=["state","scenario","value"],
        )
        .properties(height=400)
    )
    st.altair_chart(scenario_chart, use_container_width=True)

st.success(
    "Interactive tuning of weights, tiers, and outlet expansion allows stakeholders "
    "to explore different prescriptive policies and immediately see their impact."
)

