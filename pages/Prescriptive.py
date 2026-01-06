import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Pasar Mini Prescriptive Analysis", layout="wide")

st.title("Prescriptive Analysis")
st.write(
    "This page provides **actionable recommendations** for Pasar Mini based on "
    "PriceCatcher data, focusing on future food price risks, outlet access, and demand."
)

# -------------------------------------------------------------------
# 1. Load data
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
        price_df.merge(premise_df, on="premise_code", how="left")
                .merge(item_df, on="item_code", how="left")
    )
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
# 3. Staple basket selection
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
# 4. State-level metrics
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
# 5. Priority Index
# -------------------------------------------------------------------
st.subheader("Priority Index Construction")

w_access = st.slider("Weight for access (inverse outlet density)", 0.0, 1.0, 0.4, 0.05)
w_price = st.slider("Weight for price risk (avg basket price)", 0.0, 1.0, 0.3, 0.05)
w_demand = st.slider("Weight for demand (transaction volume)", 0.0, 1.0, 0.3, 0.05)

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
# 6. Visualisations (Streamlit charts)
# -------------------------------------------------------------------
st.subheader("Priority Index by State")
st.bar_chart(state_stats_sorted["priority_index"])

st.caption(
    "Higher values indicate states with fewer outlets, higher average staple pr

