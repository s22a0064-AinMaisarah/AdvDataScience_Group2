import streamlit as st
import pandas as pd

st.set_page_config(page_title="Diagnostic Analysis", layout="wide")

st.title("🔍 Diagnostic Analysis")

st.write("""
This page explains **why certain states have more Pasar Mini outlets**.

Possible diagnostic questions:
- Which states dominate Pasar Mini distribution?
- Are there regional imbalances?
- Are some states underrepresented?
""")

@st.cache_data
def load_data():
    return pd.read_csv("filtered_pasar_mini_data.csv")

df = load_data()

top_states = df['state'].value_counts().head(5)

st.subheader("Top 5 States with Highest Pasar Mini Count")
st.dataframe(top_states)
