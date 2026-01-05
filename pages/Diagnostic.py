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
    <h2 style='text-align:center;'>📊 Diagnostic Analysis Visualizations</h2>
    <p style='text-align:center; color: gray;'>
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

    # Interactive heatmap

    color_scale = ['#1f77b4', '#ff7f0e', '#e377c2', '#9467bd', '#bcbd22']  # blue, orange, pink, purple, yellow

    fig = px.imshow(
        spearman_corr,
        text_auto=".2f",
        color_continuous_scale=color_scale,
        zmin=-1,
        zmax=1,
        title="Spearman Correlation: Price vs Factors (Pasar Mini)"
    )

    # Adjust layout for title
    fig.update_layout(
        title=dict(
            text="Spearman Correlation: Price vs Factors (Pasar Mini)",
            x=0.5,          # 0 = left, 0.5 = center, 1 = right
            xanchor='center' # ensures correct positioning
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
