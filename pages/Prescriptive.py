# prescriptive_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Pasar Mini Prescriptive Analysis", layout="wide")

st.title("Pasar Mini Prescriptive Analysis")
st.write(
    "This app provides prescriptive recommendations using PriceCatcher Pasar Mini data, "
    "focusing on future food price risks, outlet access, and demand."
)

# -------------------------------------------------------------------
# 1. Load data (same URLs as in Colab)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    URL_PRICECATCHER = "https://storage.data.gov.my/pricecatcher/pricecatcher_2025-12.parquet"
    URL_PREMISE = "https://storage.data.gov.my/pricecatcher/lookup_premise.parquet"
    URL_ITEM = "https://storage.data.gov.my/pricecatcher/lookup_item.parquet"

    price_df = pd.read_parquet(URL_PRICECATCHER)
    premise_df = pd.read_parquet(URL_PREMISE)
    item_df = pd.read_parquet(URL_ITEM)

    df = price_df.merge(premise_df, on="premise_code", how="left") \
                 .merge(item_df, on="item_code", how="left")
    return df

with st.spinner("Loading PriceCatcher and lookup data..."):
    df = load_data()

st.success("Data loaded successfully.")

# -------------------------------------------------------------------
# 2. Filter Pasar Mini data
# -------------------------------------------------------------------
pm_df = df[df["premise_type"] == "Pasar Mini"].copy()

st.subheader("Pasar Mini Data Overview")
st.write("Sample of joined Pasar Mini records:")
st.dataframe(pm_df.head())

# -------------------------------------------------------------------
# 3. User selects staple basket items
# -------------------------------------------------------------------
st.subheader("Staple Basket Selection")

all_items = sorted(pm_df["item"].dropna().str.lower().unique())
default_basket = [
    "minyak masak tulen cap buruh",
    "beras cap rambutan (sst5%)",
    "bawang besar kuning/holland",
    "timun",
]

default_basket = [i for i in default_basket if i in all_items]

basket_items = st.multiselect(
    "Select staple items for the basket (used to compute average prices):",
    options=all_items,
    default=default_basket,
)

if len(basket_items) == 0:
    st.warning("Please select at least one staple item to continue.")
    st.stop()

basket = pm_df[pm_df["item"].str.lower().isin(basket_items)].copy()

# -------------------------------------------------------------------
# 4. Compute state-level metrics
# -------------------------------------------------------------------
st.subheader("State-level Metrics")

state_premises = pm_df.groupby("state")["premise"].nunique().rename("premise_count")
state_txn = pm_df.groupby("state")["price"].count().rename("txn_count")
state_basket_price = basket.groupby("state")["price"].mean().rename("avg_basket_price")

state_stats = pd.concat(
    [state_premises, state_txn, state_basket_price], axis=1
).dropna()

st.write("Raw state-level summary:")
st.dataframe(state_stats)

# -------------------------------------------------------------------
# 5. Build Priority Index
# -------------------------------------------------------------------
st.subheader("Priority Index Construction")

# slider for weights
w_access = st.slider("Weight for access (inverse outlet density)", 0.0, 1.0, 0.4, 0.05)
w_price = st.slider("Weight for price risk (avg basket price)", 0.0, 1.0, 0.3, 0.05)
w_demand = st.slider("Weight for demand (transaction volume)", 0.0, 1.0, 0.3, 0.05)

# normalisation
state_stats = state_stats.copy()
state_stats = state_stats.fillna(state_stats.median(numeric_only=True))

scaler = MinMaxScaler()

state_stats["premise_norm"] = scaler.fit_transform(
    state_stats[["premise_count"]]
)
state_stats["price_norm"] = scaler.fit_transform(
    state_stats[["avg_basket_price"]]
)
state_stats["txn_norm"] = scaler.fit_transform(
    state_stats[["txn_count"]]
)

state_stats["access_score"] = 1 - state_stats["premise_norm"]
state_stats["price_score"] = state_stats["price_norm"]
state_stats["demand_score"] = state_stats["txn_norm"]

# normalise weights to sum 1
w_sum = w_access + w_price + w_demand
if w_sum == 0:
    w_access_n = w_price_n = w_demand_n = 0
else:
    w_access_n = w_access / w_sum
    w_price_n = w_price / w_sum
    w_demand_n = w_demand / w_sum

state_stats["priority_index"] = (
    w_access_n * state_stats["access_score"]
    + w_price_n * state_stats["price_score"]
    + w_demand_n * state_stats["demand_score"]
)

state_stats_sorted = state_stats.sort_values("priority_index", ascending=False)

st.write("State Priority Table:")
st.dataframe(
    state_stats_sorted[
        [
            "premise_count",
            "txn_count",
            "avg_basket_price",
            "access_score",
            "price_score",
            "demand_score",
            "priority_index",
        ]
    ]
)

# -------------------------------------------------------------------
# 6. Plot Priority Index
# -------------------------------------------------------------------
st.subheader("Priority Index by State")

fig, ax = plt.subplots(figsize=(10, 4))
state_stats_sorted["priority_index"].plot(kind="bar", ax=ax)
ax.set_ylabel("Priority Index")
ax.set_title("Pasar Mini Intervention Priority by State")
ax.set_xticklabels(state_stats_sorted.index, rotation=90)
plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Higher values indicate states with fewer outlets, higher average staple prices, "
    "and/or stronger demand, and therefore higher priority for intervention."
)

# -------------------------------------------------------------------
# 7. Component comparison plot
# -------------------------------------------------------------------
st.subheader("Component Scores (Access, Price, Demand)")

fig2, ax2 = plt.subplots(figsize=(10, 4))
state_stats_sorted[["access_score", "price_score", "demand_score"]].plot(
    kind="bar", ax=ax2
)
ax2.set_ylabel("Normalised Score")
ax2.set_title("Components of Priority Index by State")
ax2.set_xticklabels(state_stats_sorted.index, rotation=90)
plt.tight_layout()
st.pyplot(fig2)

# -------------------------------------------------------------------
# 8. Tiering and scenario analysis
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

st.write("States with assigned tiers:")
st.dataframe(
    state_stats_sorted[
        [
            "premise_count",
            "avg_basket_price",
            "txn_count",
            "priority_index",
            "tier",
        ]
    ]
)

# Scenario: add outlets in Tier 1
N_new_outlets = st.number_input(
    "Number of additional outlets per Tier 1 state (scenario)", 0, 100, 20
)

scenario = state_stats_sorted.copy()
tier1_states = scenario[scenario["tier"] == "Tier 1"].index

scenario.loc[tier1_states, "premise_count"] += N_new_outlets

# recompute access_score and priority_index_new
scenario["premise_norm"] = scaler.fit_transform(
    scenario[["premise_count"]]
)
scenario["access_score"] = 1 - scenario["premise_norm"]

scenario["priority_index_new"] = (
    w_access_n * scenario["access_score"]
    + w_price_n * scenario["price_score"]
    + w_demand_n * scenario["demand_score"]
)

scenario["delta_priority"] = scenario["priority_index_new"] - scenario["priority_index"]

st.write("Scenario impact (negative delta = improvement):")
st.dataframe(
    scenario[["priority_index", "priority_index_new", "delta_priority", "tier"]]
    .sort_values("delta_priority")
)

fig3, ax3 = plt.subplots(figsize=(10, 4))
scenario.sort_values("priority_index", ascending=False)[
    ["priority_index", "priority_index_new"]
].plot(kind="bar", ax=ax3)
ax3.set_ylabel("Priority Index")
ax3.set_title("Before vs After Outlet Expansion Scenario")
ax3.set_xticklabels(scenario.sort_values("priority_index", ascending=False).index,
                    rotation=90)
plt.tight_layout()
st.pyplot(fig3)

st.success(
    "Use the tables and charts above to justify recommendations such as "
    "expanding Pasar Mini in Tier 1 states and improving logistics in high-priority regions."
)
