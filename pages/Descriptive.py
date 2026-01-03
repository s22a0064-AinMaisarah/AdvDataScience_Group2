import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Descriptive Analysis", layout="wide")

st.title("📊 Descriptive Analysis – Pasar Mini")

@st.cache_data
def load_data():
    return pd.read_csv("filtered_pasar_mini_data.csv")

df = load_data()

st.subheader("State Distribution")

state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']

chart = alt.Chart(state_counts).mark_bar().encode(
    x=alt.X('state:N', sort='-y', title="State"),
    y=alt.Y('count:Q', title="Number of Pasar Mini"),
    tooltip=['state', 'count']
)

st.altair_chart(chart, use_container_width=True)

st.success("This chart shows the distribution of Pasar Mini across states.")
