import streamlit as st
import altair as alt
import pandas as pd

st.set_page_config(page_title="Pasar Mini State Analysis", layout="wide")

st.title("📊 State Distribution for Pasar Mini")

# ✅ Load data locally (BEST PRACTICE)
@st.cache_data
def load_data():
    return pd.read_csv("filtered_pasar_mini_data.csv")

pasar_mini_df = load_data()

# Calculate the count of each state
state_counts_pasar_mini = (
    pasar_mini_df['state']
    .value_counts()
    .reset_index()
)

state_counts_pasar_mini.columns = ['state', 'count']

# Create Altair bar chart
chart = alt.Chart(state_counts_pasar_mini).mark_bar().encode(
    x=alt.X('state:N', sort='-y', title='State'),
    y=alt.Y('count:Q', title='Number of Entries'),
    tooltip=['state:N', 'count:Q']
).properties(
    title='State Count in Pasar Mini'
)

# Display chart
st.altair_chart(chart, use_container_width=True)

st.success(
    "Interactive bar chart displaying the count of each state in Pasar Mini premise data."
)

