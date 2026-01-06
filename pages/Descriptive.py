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

# --------------------
# Objectives Styling
# --------------------

st.markdown("""
<style>
/* Base styling for all objective boxes */
.objective-card {
    padding: 1.8rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

/* Hover effect to make it interactive */
.objective-card:hover {
    transform: translateY(-5px);
}

/* Gradient for Objective 1: Deep Blue to Purple */
.box-1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-left: 8px solid #ffffff44;
}

/* Gradient for Objective 2: Teal to Emerald */
.box-2 {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    border-left: 8px solid #ffffff44;
}

.objective-title {
    font-size: 1.4rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.objective-text {
    font-size: 1.1rem;
    line-height: 1.7;
    font-weight: 400;
    opacity: 0.95;
}
</style>
""", unsafe_allow_html=True)

st.subheader("🎯 Descriptive Analytics Objectives")

# --- Objective 1 ---
st.markdown("""
<div class="objective-card box-1">
    <div class="objective-title">📊 Objective 1: Pricing Analysis</div>
    <div class="objective-text">
        To <b>quantify</b> the current state of essential commodity pricing in Pasar Mini and identify <b>short-term historical trends</b>, while summarizing how products are distributed, sold, and priced across different regions and time periods.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Objective 2 ---
st.markdown("""
<div class="objective-card box-2">
    <div class="objective-title">📈 Objective 2: Market Distribution</div>
    <div class="objective-text">
        To provide a comprehensive overview of <b>supply chain efficiency</b> and regional price variations to better understand market volatility and consumer accessibility.
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
