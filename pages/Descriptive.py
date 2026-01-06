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

# --------------------
# Objectives
# --------------------

st.markdown("""
<style>
/* Main Container Styling */
.objective-container {
    font-family: 'Inter', sans-serif;
}

/* Base card with Glassmorphism and Gradient */
.objective-card {
    padding: 2rem;
    border-radius: 24px;
    margin-bottom: 20px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 20px -10px rgba(0, 0, 0, 0.3);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.objective-card:hover {
    transform: scale(1.02);
    box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.4);
}

/* Dynamic Gradients */
/* Box 1: Electric Violet to Royal Blue */
.box-1 {
    background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%);
}

/* Box 2: Sunset Orange to Deep Red */
.box-2 {
    background: linear-gradient(135deg, #FF512F 0%, #DD2476 100%);
}

/* Box 3: Ocean Teal to Bright Green (Optional) */
.box-3 {
    background: linear-gradient(135deg, #02AAB0 0%, #00CDAC 100%);
}

/* Typography Enhancements */
.objective-title {
    font-size: 1.5rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.objective-subtitle {
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 4px;
    letter-spacing: 2px;
}

.objective-text {
    font-size: 1.15rem;
    line-height: 1.6;
    font-weight: 300;
    opacity: 0.95;
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 12px;
    border-left: 4px solid rgba(255, 255, 255, 0.5);
}

/* Icon styling */
.icon {
    margin-right: 12px;
    font-size: 1.8rem;
}
</style>
""", unsafe_allow_html=True)

st.subheader("🎯 Descriptive Analytics Objectives")

# --- Objective 1 ---
st.markdown("""
<div class="objective-card box-1">
    <div class="objective-subtitle">Primary Goal</div>
    <div class="objective-title"><span class="icon">📊</span> Pricing Analysis</div>
    <div class="objective-text">
        To descriptively <b>analyse price patterns</b>, distribution characteristics, and category-based variations of items across <b>time, location,</b> and item classifications.
    </div>
</div>
""", unsafe_allow_html=True)

# --- Objective 2 (Example of adding a second one) ---
st.markdown("""
<div class="objective-card box-2">
    <div class="objective-subtitle">Secondary Goal</div>
    <div class="objective-title"><span class="icon">📍</span> Spatial Trends</div>
    <div class="objective-text">
        To identify <b>geographical outliers</b> and regional price disparities to ensure market transparency and consumer protection across all Mini Markets.
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
