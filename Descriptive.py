import streamlit as st
import pandas as pd
import altair as alt

st.title("📊 Descriptive Analysis – Pasar Mini")

@st.cache_data
def load_data():
    return pd.read_csv("filtered_pasar_mini_data.csv")

df = load_data()

state_counts = df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']

chart = alt.Chart(state_counts).mark_bar().encode(
    x=alt.X('state:N', sort='-y', title='State'),
    y=alt.Y('count:Q', title='Number of Entries'),
    tooltip=['state:N', 'count:Q']
).properties(
    title="State Distribution of Pasar Mini"
)

st.altair_chart(chart, use_container_width=True)
