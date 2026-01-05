import streamlit as st
import pandas as pd

# --------------------
# Load CSV from GitHub Raw
# --------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/refs/heads/main/dataset/pasar_mini_data_updated.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# --------------------
# Streamlit UI
# --------------------
st.title("📊 Pasar Mini Diagnostic Analytics Dashboard")

st.write("### Preview of the Dataset")
st.dataframe(df.head())

st.write("### Dataset Summary")
st.write(df.describe(include='all'))

