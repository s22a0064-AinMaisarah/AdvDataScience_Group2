import streamlit as st
import pandas as pd
import plotly.express as px

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
    '<div class="center-title">📊 Pasar Mini Diagnostic Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Diagnostic Analysis of Price Patterns</div>',
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

# --------------------
# Summary Box
# --------------------

import streamlit as st

# --- Calculate Diagnostic Metrics ---
total_records = pasar_mini_df.shape[0]  # total rows
unique_items = pasar_mini_df['item_enc'].nunique()  # total unique items
unique_categories = pasar_mini_df['item_category_enc'].nunique()  # unique item categories
average_price = round(pasar_mini_df['price'].mean(), 2)  # average price

# --- Metric Setup ---
metrics = [
    ("Total Records", total_records, "Total number of Pasar Mini price records in the dataset."),
    ("Unique Items", unique_items, "Number of distinct items sold in Pasar Mini."),
    ("Unique Item Categories", unique_categories, "Number of distinct item categories."),
    ("Average Price (RM)", average_price, "Mean price across all records in Pasar Mini.")
]

# --- Display Metrics in 4 Columns ---
cols = st.columns(4)
for col, (label, value, help_text) in zip(cols, metrics):
    col.markdown(f"""
        <div style="
            background-color:#F8F9FA; 
            border:1px solid #DDD; 
            border-radius:10px; 
            padding:15px; 
            text-align:center;
            min-height:120px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        ">
            <div style="font-size:16px; font-weight:700; color:#1E293B; margin-bottom:8px; line-height:1.2em;">
                {label} <span title="{help_text}" style="cursor:help; color:#2563EB;">🛈</span>
            </div>
            <div style="font-size:26px; font-weight:800; color:#111;">{value}</div>
        </div>
    """, unsafe_allow_html=True)


st.markdown("---")

# --------------------
# Objectives
# --------------------

st.markdown("""
<style>
.objective-box {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #1f77b4, #6baed6);
    color: white;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    margin-bottom: 1.5rem;
}
.objective-box-2 {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #b22222, #ef8a62);
    color: white;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}
.objective-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.objective-text {
    font-size: 1.05rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.subheader("🎯 Diagnostic Analytics Objectives")

# Objective 1
st.markdown("""
<div class="objective-box">
    <div class="objective-title">Objective 1: Price Driver Identification</div>
    <div class="objective-text">
        To identify and quantify the key factors driving price variations in Pasar Mini 
        by applying correlation analysis, statistical testing, and regression techniques 
        across item, premise, and geographical dimensions.
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

# -----------------------------
# Spearman Correlation Analysis
# -----------------------------

with st.container():
    st.subheader("🔗 Correlation Analysis")
    st.caption(
        "Spearman correlation is applied to examine monotonic relationships "
        "between price and encoded categorical factors in Pasar Mini."
    )

    # Select relevant columns
    corr_cols = [
        'price',
        'state_enc',
        'district_enc',
        'item_group_enc',
        'item_category_enc'
    ]

    # Compute Spearman correlation
    spearman_corr = pasar_mini_df[corr_cols].corr(method='spearman')

    
    # Interactive heatmap with cool-warm palette

    fig = px.imshow(
    spearman_corr,
    text_auto=".2f",
    color_continuous_scale=[
        "#d73027",  # strong negative
        "#f46d43",  # moderate negative
        "#ffffbf",  # neutral / zero
        "#66bd63",  # moderate positive
        "#1a9850"   # strong positive
    ],
    zmin=-1,
    zmax=1,
    title="Spearman Correlation: Price vs Factors (Pasar Mini)"
    )

    # Adjust title position
    
    fig.update_layout(
    title=dict(
        text="Spearman Correlation: Price vs Factors (Pasar Mini)",
        x=0.5,         # 0 = left, 0.5 = center
        xanchor='center'
    ),
    coloraxis_colorbar=dict(title="Correlation"),
    margin=dict(l=40, r=40, t=80, b=40)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Optional: show correlation table
    with st.expander("📄 View Correlation Matrix (Table)", expanded=False):
        st.dataframe(spearman_corr, use_container_width=True)





# Objective 2
st.markdown("""
<div class="objective-box-2">
    <div class="objective-title">Objective 2: Root Cause & Trend Investigation</div>
    <div class="objective-text">
        To investigate the underlying root causes and temporal patterns behind observed 
        price trends and anomalies in Pasar Mini through segmentation, drill-down analysis, 
        and root cause decomposition techniques.
    </div>
</div>
""", unsafe_allow_html=True)
