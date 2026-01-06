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
# Header & Divider 
# --------------------

st.markdown("""
<style>
/* Main Title Styling */
.center-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

/* Subtitle Styling */
.subtitle {
    text-align: center;
    font-size: 1rem;
    color: #666;
    font-family: 'Inter', sans-serif;
    letter-spacing: 1px;
    margin-bottom: 1rem;
}

/* Modern Gradient Divider */
.divider {
    height: 3px;
    background: linear-gradient(90deg, transparent, #4facfe, #764ba2, transparent);
    margin: 10px auto 30px auto;
    width: 80%;
    border-radius: 50%;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Header Section
# --------------------

st.markdown(
    '<div class="center-title">Descriptive Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Nurul Ain Maisarah Binti Hamidin | S22A0064</div>',
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
st.markdown("""
<style>
/* Make the expander header look like a clickable button */
.stExpander {
    background-color: #1E1E1E !important; /* Dark background to match your theme */
    border: 2px solid #4facfe !important; /* Bright blue border */
    border-radius: 15px !important;
}

/* Style the title text inside the expander */
.stExpander summary {
    color: #4facfe !important; /* Bright blue text */
    font-weight: bold !important;
    font-size: 1.1rem !important;
}

/* Add a glowing pulse effect to tell users 'Click Me' */
@keyframes glow {
    0% { box-shadow: 0 0 5px #4facfe; }
    50% { box-shadow: 0 0 20px #00f2fe; }
    100% { box-shadow: 0 0 5px #4facfe; }
}

.stExpander {
    animation: glow 3s infinite;
}

/* Change color when user hovers */
.stExpander:hover {
    background-color: #252525 !important;
    transform: scale(1.01);
    transition: 0.3s;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Implementation with Attention-Grabbing Arrow
# --------------------

# We use 🔽 to show it can be opened and 🔎 for the action
with st.expander("🔽 🔎 CLICK TO REVEAL DATASET PREVIEW", expanded=False):
    st.write("### Previewing First 5 Rows")
    st.dataframe(pasar_mini_df.head(), use_container_width=True)
# --- Display metrics in 4 columns ---
# ---------------------------------------------------------
# KPI METRICS
# --------------------------------------------------------
st.markdown("""
<style>
/* Container for the metrics */
.metric-container {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}

/* Stylish Card Design */
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 15px;
    flex: 1;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border-bottom: 4px solid #ddd; /* Placeholder color */
    transition: transform 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}

/* Individual accent colors */
.m-max { border-color: #FF4B4B; } /* Red */
.m-min { border-color: #00CC96; } /* Green */
.m-top { border-color: #636EFA; } /* Blue */
.m-cat { border-color: #AB63FA; } /* Purple */

.metric-label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 1.25rem;
    font-weight: 800;
    color: #31333F;
}

.metric-help {
    font-size: 0.75rem;
    color: #999;
    margin-top: 8px;
    font-style: italic;
    line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

st.markdown("### Key Dataset Metrics")

# Use columns to lay out the custom cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card m-max">
        <div class="metric-label">Max Price</div>
        <div class="metric-value">RM 498.00</div>
        <div class="metric-help">Bawang Besar Import (India) (1kg)<br>2025-12-19</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card m-min">
        <div class="metric-label">Min Price</div>
        <div class="metric-value">RM 0.50</div>
        <div class="metric-help">Serbuk Kari Adabi<br>2025-12-08</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card m-top">
        <div class="metric-label">Top Premise</div>
        <div class="metric-value">1,641</div>
        <div class="metric-help">Kifarah Fresh Mart</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card m-cat">
        <div class="metric-label">Top Item Category</div>
        <div class="metric-value">67,098</div>
        <div class="metric-help">Barangan Berbungkus</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------
# Objectives 
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

st.subheader("Decriptive Objectives")

# --- Objective ---
st.markdown("""
<div class="tiny-card bg-purple">
    <div class="tiny-text">
        To descriptively price patterns, distribution characteristics, across time, location, and item classifications among Pasar Mini.
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
