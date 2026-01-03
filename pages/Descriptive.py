import streamlit as st
import pandas as pd
import altair as alt

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crime Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📊 Crime Analytics Dashboard Menu")
    st.write("Gain insights into relationships between socioeconomic factors and crime patterns across cities.")
    st.markdown("---")
    st.subheader("📂 Navigation")
    st.info("Use the menu to explore different analysis modules.")
    st.markdown("---")
    st.caption("👩🏻‍💻 Created by **Nurul Ain Maisarah Hamidin (2025)** | Scientific Visualization Project 🌟")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🚨 Crime Pattern Clustering & PCA Dashboard")
st.markdown("""
### 🎯 Objective  
The objective of this visualization is to identify **patterns in urban crime** by grouping similar crime profiles.  
This helps reveal hidden patterns across regions and demographics — guiding urban safety strategies.
""")


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
