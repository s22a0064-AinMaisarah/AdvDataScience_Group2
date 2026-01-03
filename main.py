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
page = st.sidebar.radio(
    "Navigate",
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

if page == "Descriptive":
    Descriptive.app()

elif page == "Diagnostic":
    Diagnostic.app()

elif page == "Predictive":
    Predictive.app()

elif page == "Prescriptive":
    Prescriptive.app()
