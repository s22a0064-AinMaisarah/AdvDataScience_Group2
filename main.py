import streamlit as st
import pandas as pd
import altair as alt

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Pasar Mini Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Load Data (Cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("filtered_pasar_mini_data.csv")

df = load_data()

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📊 Pasar Mini State Analysis Dashboard")

st.markdown(
    """
    This dashboard visualizes the distribution of **Pasar Mini** data
    across Malaysian states based on the uploaded dataset.
    """
)

# --------------------------------------------------
# State Distribution Chart
# --------------------------------------------------
state_counts = (
    df["state"]
    .value_counts()
    .reset_index()
)

state_counts.columns = ["state", "count"]

chart = alt.Chart(state_counts).mark_bar().encode(
    x=alt.X("state:N", sort="-y", title="State"),
    y=alt.Y("count:Q", title="Number of Records"),
    tooltip=["state:N", "count:Q"]
).properties(
    title="Number of Pasar Mini Records by State",
    height=500
)

st.altair_chart(chart, use_container_width=True)

# --------------------------------------------------
# Dataset Summary
# --------------------------------------------------
with st.expander("📌 Dataset Summary"):
    st.write(f"Total records: **{df.shape[0]:,}**")
    st.write(f"Total columns: **{df.shape[1]}**")
    st.write("Columns:")
    st.write(list(df.columns))
