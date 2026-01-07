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
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/refs/heads/main/dataset/pasar_mini_data.csv"
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
    background: linear-gradient(135deg, #b22222, #ef8a62);
    color: white;
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
    margin-bottom: 1.5rem;
}
.objective-box-2 {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #1f77b4, #6baed6);
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
    title="Spearman Correlation: Price vs Factors (Pasar Mini-Numerical)"
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
    title="Pearson Correlation: Price vs Time (Pasar Mini-Categorical)"
)

fig.update_layout(
    title=dict(x=0.5, xanchor="center"),
    coloraxis_colorbar=dict(title="Correlation")
)


# Optional: show correlation table
with st.expander("📄 View Correlation Matrix", expanded=False):
    st.dataframe(spearman_corr, use_container_width=True)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Regression Analysis
# -----------------------------

with st.container():
    st.subheader("📈 Multiple Regression: Key Drivers of Price")

    st.caption(
        "A multiple linear regression model is applied to examine how geographical "
        "and product-related factors jointly influence price variations in Pasar Mini."
    )

    # Independent variables
    X_multi = pasar_mini_df[
        ['state_enc', 'district_enc', 'item_group_enc', 'item_category_enc']
    ]

    X_multi = sm.add_constant(X_multi)
    y = pasar_mini_df['price']

    # Fit model
    multi_reg = sm.OLS(y, X_multi).fit()


    col1, col2 = st.columns(2)

    col1.metric(
        label="R-squared",
        value=round(multi_reg.rsquared, 3),
        help="Proportion of price variation explained by the model."
    )

    col2.metric(
        label="F-statistic",
        value=round(multi_reg.fvalue, 2),
        help="Tests whether the model is statistically significant overall."
    )
    fig = px.scatter(
        pasar_mini_df,
        x='item_category_enc',
        y='price',
        trendline='ols',
        trendline_color_override='red',
        color_discrete_sequence=['darkblue'],
        title='Price vs Item Category (Key Driver – Pasar Mini)',
        labels={
            'item_category_enc': 'Item Category',
            'price': 'Price (RM)'
        }
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 View Full Regression Summary"):
        st.text(multi_reg.summary())

    st.markdown(
        """
        <div style="
            background-color:#FFF7ED;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
        ">
        <b>Interpretation:</b><br>
        The regression results indicate that price variations are influenced by a combination of 
        geographical and product-related factors. The R-squared value suggests that the selected 
        variables explain a meaningful proportion of price variability in Pasar Mini. Overall, the 
        significant F-statistic confirms that the model is suitable for identifying key drivers 
        behind observed pricing patterns.
        </div>
        """,
        unsafe_allow_html=True
    )



# -----------------------------
# Statistical Testing 
# -----------------------------

# --- Perform ANOVA on price across item categories ---
anova_model = ols('price ~ C(item_category_enc)', data=pasar_mini_df).fit()
anova_result = sm.stats.anova_lm(anova_model, typ=2)

# --- Boxplot of price by item category ---
st.subheader("📦 Price Distribution by Item Category")
st.caption("Visual representation of price spread for each item category.")

fig = px.box(
    pasar_mini_df,
    x='item_category_enc',
    y='price',
    color='item_category_enc',
    color_discrete_sequence=px.colors.diverging.RdBu,
    title='Price Distribution by Item Category (Pasar Mini)',
    labels={
        'item_category_enc': 'Item Category',
        'price': 'Price (RM)'
    }
)

# --- Display ANOVA result in Streamlit ---
st.subheader("📊 ANOVA: Price Differences Across Item Categories")
st.caption(" ANOVA test examines whether the average prices differ significantly "
    "between item categories in Pasar Mini."
) 

st.dataframe(anova_result, use_container_width=True)    

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)



st.subheader("🔍 Chi-Square Test: State vs Item Category")

st.caption(
    "This analysis examines whether the distribution of item categories "
    "is independent of state using the Chi-Square Test of Independence."
)

# Create contingency table
contingency_table = pd.crosstab(
    pasar_mini_df['state'],
    pasar_mini_df['item_category']
)

# Display contingency table
st.markdown("### 📊 Contingency Table")
st.dataframe(contingency_table, use_container_width=True)

# Perform Chi-Square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# Display statistical results
st.markdown("### 📈 Chi-Square Test Results")

col1, col2, col3 = st.columns(3)
col1.metric("Chi-Square Statistic", round(chi2, 2))
col2.metric("Degrees of Freedom", dof)
col3.metric("p-value", round(p_value, 5))

st.markdown("### 🧠 Chi-Square Interpretation")

if p_value < 0.05:
    st.success(
        "The p-value is less than 0.05, indicating a statistically significant "
        "association between state and item category. This suggests that item "
        "distribution patterns vary across states."
    )
else:
    st.info(
        "The p-value is greater than 0.05, indicating no statistically significant "
        "association between state and item category."
    )


st.markdown("---")

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
# Segmentation Analysis 
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
        title='Average Price by State and Item Group (Pasar Mini-Numerical)',
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


# --- Aggregate by item category ---
seg_category = (
    pasar_mini_df.groupby('item_category_enc')['price']
    .agg(['mean', 'count'])
    .reset_index()
    .sort_values('count', ascending=False)
)

# --- Plotly bar chart ---
fig = px.bar(
    seg_category,
    x='item_category_enc',
    y='mean',
    title='Average Price by Item Category (Pasar Mini-Categorical)',
    color_discrete_sequence=['darkblue'],  # single color
    labels={
        'item_category_enc': 'Item Category',
        'mean': 'Average Price (RM)'
    }
)

# --- Display chart in Streamlit ---
st.plotly_chart(fig, use_container_width=True)

# --- Optional: Show top categories in an expander ---
with st.expander("📋 Top 10 Item Categories by Average Price"):
    st.dataframe(
        seg_category.head(10),
        use_container_width=True
    )

# -----------------------------
# Drill-Down Analysis 
# -----------------------------

st.subheader("🔎 Drill-Down Analysis: Top Items in Johor (Pasar Mini)")

st.caption(
    "This analysis examines whether the distribution of item categories "
    "is independent of state using the Chi-Square Test of Independence."
)

# Drill-down for Johor
drill_johor = (
    pasar_mini_df[pasar_mini_df['state'] == 'JOHOR']
    .groupby(['district', 'item'])['price']
    .agg(['mean', 'count'])
    .reset_index()
    .sort_values('count', ascending=False)
)

fig = px.bar(
    drill_johor.head(10),
    x='item',
    y='count',
    title='Top Items in Johor by Transaction Volume (Pasar Mini)',
    labels={
        'item': 'Item',
        'count': 'Number of Records'
    },
    color_discrete_sequence=['#1E3A8A']  # Deep blue
)

st.plotly_chart(fig, use_container_width=True)

# Dropdown to display top 10 items
with st.expander("📋 View Top 10 Items by Transaction Count (Johor)"):
    st.dataframe(
        drill_johor.head(10),
        use_container_width=True
    )

with st.container():
    st.markdown(
        """
        <div style="
            background-color: #FFF1E6;
            border-left: 6px solid #FB923C;
            padding: 18px 22px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        ">
            <h4 style="margin-top: 0; color: #7C2D12;">
                🧠 Diagnostic Interpretation
            </h4>
            <p style="font-size: 15px; color: #3F2A1D;">
                The results indicate that a small number of items dominate transaction volumes in Johor, 
                suggesting strong and consistent consumer demand for essential goods. 
                This concentration reflects localized consumption patterns influenced by daily necessities 
                rather than price variability alone. 
                Such trends highlight how regional purchasing behavior acts as a key driver of observed price 
                stability and frequency in Pasar Mini markets.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

