import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Pasar Mini Prescriptive Analysis", layout="wide")

st.title("Prescriptive Analysis – Pasar Mini")
st.write(
    "This page provides data-driven recommendations for Pasar Mini outlet expansion "
    "and logistics support using the PriceCatcher dataset (December 2025)."
)
st.subheader("Objectives of Prescriptive Analysis")

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

# -------------------------------------------------------------------
# 1. Load and join data
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    URL_PRICECATCHER = "https://storage.data.gov.my/pricecatcher/pricecatcher_2025-12.parquet"
    URL_PREMISE = "https://storage.data.gov.my/pricecatcher/lookup_premise.parquet"
    URL_ITEM = "https://storage.data.gov.my/pricecatcher/lookup_item.parquet"

    price_df = pd.read_parquet(URL_PRICECATCHER)
    premise_df = pd.read_parquet(URL_PREMISE)
    item_df = pd.read_parquet(URL_ITEM)

    df = (
        price_df.merge(premise_df, on='premise_code', how='left')
                .merge(item_df, on='item_code', how='left')
    )
    df["date"] = pd.to_datetime(df["date"])
    return df

with st.spinner("Loading PriceCatcher and lookup tables..."):
    df = load_data()

pm_df = df[df["premise_type"] == "Pasar Mini"].copy()

st.subheader("Pasar Mini Data Sample")
st.dataframe(pm_df.head())

# -------------------------------------------------------------------
# 2. Staple basket selection
# -------------------------------------------------------------------
st.subheader("Staple Basket for Price Risk")

all_items = sorted(pm_df["item"].dropna().str.lower().unique())
default_basket = [
    "minyak masak tulen cap buruh",
    "beras cap rambutan (sst5%)",
    "bawang besar kuning/holland",
    "timun",
]
default_basket = [i for i in default_basket if i in all_items]

basket_items = st.multiselect(
    "Select staple items (used to compute average basket price):",
    options=all_items,
    default=default_basket
)

if len(basket_items) == 0:
    st.warning("Please select at least one staple item.")
    st.stop()

pm_df["item_lower"] = pm_df["item"].str.lower()
basket_df = pm_df[pm_df["item_lower"].isin(basket_items)].copy()

# -------------------------------------------------------------------
# 3. State-level metrics
# -------------------------------------------------------------------
st.subheader("State-Level Metrics")

state_premises = pm_df.groupby("state")["premise"].nunique().rename("premise_count")
state_txn = pm_df.groupby("state")["price"].count().rename("txn_count")
state_basket_price = basket_df.groupby("state")["price"].mean().rename("avg_basket_price")

state_stats = pd.concat(
    [state_premises, state_txn, state_basket_price], axis=1
).dropna()

st.write("Summary table:")
st.dataframe(state_stats)

# -------------------------------------------------------------------
# 4. Priority Index construction
# -------------------------------------------------------------------
st.subheader("Priority Index Construction")

w_access = st.slider("Weight: Access (inverse outlet density)", 0.0, 1.0, 0.4, 0.05)
w_price  = st.slider("Weight: Price risk (avg basket price)",   0.0, 1.0, 0.3, 0.05)
w_demand = st.slider("Weight: Demand (transaction volume)",    0.0, 1.0, 0.3, 0.05)

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
if w_sum == 0:
    w_access_n = w_price_n = w_demand_n = 0
else:
    w_access_n = w_access / w_sum
    w_price_n  = w_price  / w_sum
    w_demand_n = w_demand / w_sum

state_stats["priority_index"] = (
    w_access_n * state_stats["access_score"]
    + w_price_n * state_stats["price_score"]
    + w_demand_n * state_stats["demand_score"]
)

state_stats_sorted = state_stats.sort_values("priority_index", ascending=False)

st.write("Priority table (higher = higher intervention priority):")
st.dataframe(
    state_stats_sorted[
        ["premise_count","txn_count","avg_basket_price",
         "access_score","price_score","demand_score","priority_index"]
    ]
)

# -------------------------------------------------------------------
# 5. Visualisations
# -------------------------------------------------------------------
st.subheader("Priority Index by State")
st.bar_chart(state_stats_sorted["priority_index"])

st.caption(
    "States with higher Priority Index have fewer Pasar Mini outlets, higher staple prices, "
    "and/or stronger demand, and are therefore higher priority for intervention."
)

st.subheader("Component Scores (Access, Price, Demand)")
st.bar_chart(
    state_stats_sorted[["access_score","price_score","demand_score"]]
)

# -------------------------------------------------------------------
# 6. Tiering and scenario analysis
# -------------------------------------------------------------------
st.subheader("Tiers and Outlet Expansion Scenario")

top_n_tier1 = st.number_input("Number of Tier 1 states", 1, 8, 5)
top_n_tier2 = st.number_input("Number of Tier 2 states", 1, 8, 5)

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

st.write("States and tiers:")
st.dataframe(
    state_stats_sorted[
        ["premise_count","avg_basket_price","txn_count","priority_index","tier"]
    ]
)

N_new_outlets = st.number_input(
    "Additional outlets per Tier 1 state (scenario)", 0, 100, 20
)

scenario = state_stats_sorted.copy()
tier1_states = scenario[scenario["tier"] == "Tier 1"].index
scenario.loc[tier1_states, "premise_count"] += N_new_outlets

# recompute access and priority_index_new
scenario["premise_norm"] = scaler.fit_transform(scenario[["premise_count"]])
scenario["access_score"] = 1 - scenario["premise_norm"]

scenario["priority_index_new"] = (
    w_access_n * scenario["access_score"]
    + w_price_n * scenario["price_score"]
    + w_demand_n * scenario["demand_score"]
)

scenario["delta_priority"] = scenario["priority_index_new"] - scenario["priority_index"]

st.write("Scenario impact (negative delta = improvement):")
st.dataframe(
    scenario[["priority_index","priority_index_new","delta_priority","tier"]]
    .sort_values("delta_priority")
)

st.subheader("Before vs After Outlet Expansion Scenario")
scenario_sorted = scenario.sort_values("priority_index", ascending=False)[
    ["priority_index","priority_index_new"]
]
st.bar_chart(scenario_sorted)

st.success(
    "The Priority Index and scenario results above justify recommendations such as "
    "expanding Pasar Mini in Tier 1 states and focusing on logistics optimisation in Tier 2."
)

