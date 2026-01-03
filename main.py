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

