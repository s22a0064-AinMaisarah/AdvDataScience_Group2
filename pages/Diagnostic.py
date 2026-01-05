import streamlit as st
import pandas as pd
import plotly.express as px
import html

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

    # 1. Compute Spearman correlation
    spearman_corr = pasar_mini_df[corr_cols].corr(method='spearman')

    
    # Interactive heatmap with cool-warm palette

    fig = px.imshow(
    spearman_corr,
    text_auto=".2f",
    color_continuous_scale=[
        "#313695",  # strong negative (dark blue)
        "#74add1",  # moderate negative (light blue)
        "#ffffbf",  # neutral / zero (pale yellow)
        "#f46d43",  # moderate positive (orange-red)
        "#d73027"   # strong positive (dark red)
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
    with st.expander("📄 View Correlation Matrix", expanded=False):
        st.dataframe(spearman_corr, use_container_width=True)


# 2. Calculate Pearson correlation

# Ensure date is datetime
pasar_mini_df['date'] = pd.to_datetime(pasar_mini_df['date'])

# Convert date to numeric (ordinal)
pasar_mini_df['date_numeric'] = pasar_mini_df['date'].map(pd.Timestamp.toordinal)

# Calculate Pearson correlation
pearson_corr = pasar_mini_df[['price', 'date_numeric']].corr(method='pearson')

# Plot heatmap with your red-positive / blue-negative palette
fig = px.imshow(
    pearson_corr,
    text_auto=".2f",
    color_continuous_scale=[
        "#313695",  # strong negative (blue)
        "#74add1",  # moderate negative
        "#ffffbf",  # neutral
        "#f46d43",  # moderate positive
        "#d73027"   # strong positive (red)
    ],
    zmin=-1,
    zmax=1,
    title="Pearson Correlation: Price vs Time (Pasar Mini)"
)

fig.update_layout(
    title=dict(x=0.5, xanchor="center"),
    coloraxis_colorbar=dict(title="Correlation")
)

st.plotly_chart(fig, use_container_width=True)





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


# -----------------------------
# Segmentation Analysis (With Filters)
# -----------------------------

with st.container():
    st.subheader("🧩 Market Segmentation by State and Item Group")
    st.caption(
        "This segmentation groups Pasar Mini data by state and item group, "
        "and computes the average price to identify price differences "
        "across regional and product-based segments."
    )

    # Aggregate average price
    seg_state_item = (
        pasar_mini_df
        .groupby(['state_enc', 'item_group_enc'], as_index=False)['price']
        .mean()
        .sort_values('price', ascending=False)
    )

    # Bar chart visualization
    fig = px.bar(
        seg_state_item,
        x='state_enc',
        y='price',
        color='item_group_enc',
        barmode='group',
        title='Average Price by State and Item Group (Pasar Mini)',
        labels={
            'state_enc': 'State',
            'price': 'Average Price',
            'item_group_enc': 'Item Group'
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Create mapping from raw dataset ---
state_mapping = (
    pasar_mini_df[['state_enc', 'state']]
    .drop_duplicates()
    .sort_values('state_enc')
    .set_index('state_enc')['state']
    .to_dict()
)

# --- Add state column to segmentation table ---
seg_state_item['state'] = seg_state_item['state_enc'].map(state_mapping)

# --- Streamlit Expander with State Dropdown ---
with st.expander("📋 View Top 10 State–Item Group Segments by Average Price"):

    # State dropdown
    selected_state = st.selectbox(
        "Select State",
        options=sorted(seg_state_item['state'].dropna().unique())
    )

    # Filter by selected state
    filtered_seg = seg_state_item[
        seg_state_item['state'] == selected_state
    ]

    # Display table with both state_enc and state
    st.dataframe(
        filtered_seg[['state_enc', 'state', 'item_group_enc', 'price']].head(10),
        use_container_width=True
    )

