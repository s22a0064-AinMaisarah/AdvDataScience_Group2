import streamlit as st
import pandas as pd
import plotly.express as px
import html
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as stats


# --------------------
# Load CSV from GitHub Raw
# --------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/refs/heads/main/dataset/pasar_mini_data_updated.csv"
    df = pd.read_csv(url)
    return df

pasar_mini_df = load_data()

# --------------------
# Streamlit UI
# --------------------
st.markdown(
    '<div class="center-title"> Pasar Mini Decriptive Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Descriptive Analysis of Price Patterns</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.center-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #6c757d;
    margin-bottom: 1.2rem;
}
.divider {
    border-top: 3px solid #1f77b4;
    margin: 1.2rem 0 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Dataset Summary
# --------------------

with st.expander("🔍 Preview of the Dataset", expanded=False):
    st.dataframe(pasar_mini_df.head(), use_container_width=True)

with st.expander("📈 Dataset Summary Statistics (Numeric Columns)", expanded=False):
    numeric_df = pasar_mini_df.select_dtypes(include=['int64', 'float64'])
    st.dataframe(
        numeric_df.describe(),
        use_container_width=True
    )

# -----------------------------
# Diagnostic Summary Metrics
# -----------------------------
total_records = pasar_mini_df.shape[0]
unique_items = pasar_mini_df['item_enc'].nunique()
unique_categories = pasar_mini_df['item_category_enc'].nunique()
average_price = round(pasar_mini_df['price'].mean(), 2)

metrics = [
    ("Total Records", total_records, "Total number of price observations collected from the raw Pasar Mini dataset."),
    ("Unique Items", unique_items, "Number of distinct items in the raw dataset."),
    ("Item Categories", unique_categories, "Number of unique product categories in the raw dataset."),
    ("Average Price (RM)", average_price, "Overall mean price calculated from the raw dataset.")
]

# --- Display metrics in 4 columns ---
cols = st.columns(4)

for col, (label, value, info) in zip(cols, metrics):
    safe_info = html.escape(info)

    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #A78BFA, #F472B6, #FACC15, #3B82F6);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
            min-height: 130px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.12);
            color: #FFFFFF;
        ">
        <div style="
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 6px;
        ">
        {label}
        <span 
        title="{safe_info}"
        style="
        margin-left: 6px;
        font-weight: 800;
        color: #FFFFFF;
        cursor: help;
        border-radius: 50%;
        padding: 2px 6px;
        background-color: rgba(255,255,255,0.3);
        "
        >?</span>
        </div>
        <div style="
        font-size: 28px;
        font-weight: 800;
        ">
        {value}
        </div>
        </div>
        """,
        unsafe_allow_html=True
        )

st.markdown("---")

import streamlit as st

import streamlit as st

import streamlit as st

# --------------------
# Compact Objectives Styling
# --------------------

st.markdown("""
<style>
/* Smaller, sleek card container */
.tiny-card {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.tiny-card:hover {
    transform: translateX(5px); /* Slides slightly right instead of growing */
}

/* Compact Gradients */
.bg-blue { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); }
.bg-purple { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
.bg-orange { background: linear-gradient(90deg, #f6d365 0%, #fda085 100%); }

.tiny-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
}

.tiny-text {
    font-size: 0.9rem;
    line-height: 1.4;
    font-weight: 400;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

st.subheader("🎯 Project Objectives")

# --- Objective 1 ---
st.markdown("""
<div class="tiny-card bg-purple">
    <div class="tiny-title">📊 Objective 1: Pricing Analysis</div>
    <div class="tiny-text">
        Analyse price patterns, distribution, and category variations across time and location.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Objective 2 ---
st.markdown("""
<div class="tiny-card bg-blue">
    <div class="tiny-title">📍 Objective 2: Spatial Trends</div>
    <div class="tiny-text">
        Identify geographical price disparities and market accessibility across regions.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Objective 3 ---
st.markdown("""
<div class="tiny-card bg-orange">
    <div class="tiny-title">📈 Objective 3: Historical Flow</div>
    <div class="tiny-text">
        Summarize short-term historical trends for essential commodity distribution.
    </div>
</div>
""", unsafe_allow_html=True)
# --------------------
# Visualization Sections
# --------------------

st.markdown("---")

st.markdown(
    """
    <h2 style='text-align:left;'>📊 Diagnostic Analysis Visualizations</h2>
    <p style='text-align:left; color: gray;'>
    This section presents visual evidence supporting the diagnostic objectives,
    including correlation analysis, segmentation, statistical testing, and root cause analysis.
    </p>
    """,
    unsafe_allow_html=True
)
