# =========================================================
# Descriptive Across Price Among Pasar Mini
# Structured Version — by Nurul Ain Maisarah Hamidin (2026)
# =========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Descriptive Across Price Among Pasar Mini",
    layout="wide"
)

# --------------------
# LOAD DATA
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/pasar_mini_data_updated.csv")  # local path
    df['date'] = pd.to_datetime(df['date'])
    return df

pasar_mini_df = load_data()

# ---------------------------------------------------------
# 3. CUSTOM STYLES (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    .center-title {
        text-align: center; font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem; letter-spacing: -1px;
    }
    .subtitle {
        text-align: center; font-size: 1rem; color: #666;
        font-family: 'Inter', sans-serif; letter-spacing: 1px; margin-bottom: 1rem;
    }
    .divider {
        height: 3px; background: linear-gradient(90deg, transparent, #4facfe, #764ba2, transparent);
        margin: 10px auto 30px auto; width: 80%; border-radius: 50%;
    }
    .metric-card {
        background: #ffffff; border-radius: 12px; padding: 15px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-bottom: 4px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. HEADER SECTION
# ---------------------------------------------------------
st.markdown('<div class="center-title">Descriptive Across Price Among Pasar Mini</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Nurul Ain Maisarah Binti Hamidin | S22A0064</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DATA VISUALIZATION TABLE
# ---------------------------------------------------------
st.subheader("Disagreement Count Matrix")
st.dataframe(disagree_area_type_original, use_container_width=True)

# ---------------------------------------------------------
# 6. KPI METRICS & INSIGHTS
# ---------------------------------------------------------
    st.write("### Descriptive Summary")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Lowest Price Item",
            value="RM 0.50",
            help="Serbuk Kari Ayam dan Daging Adabi."
        )

    with m_col2:
        st.metric(
            label="Highest Price Item",
            value="RM 498.00",
            help="Bawang Besar Import (India) | 2025-12-19."
        )

    with m_col3:
        st.metric(
            label="Most Sales Item",
            value="minyak masak tulen cap buruh",
            help="Average Price RM 18.77."
        )

    with m_col4:
        st.metric(
            label="Most Sales Premise",
            value="1641",
            help="Pasar Raya Kifarah Fresh Mart."
        )

    st.markdown("---")
