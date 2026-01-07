import streamlit as st
import pandas as pd
import plotly.express as px
import html
from statsmodels.formula.api import ols
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go



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

# Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF7ED;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
        ">
        <b>Interpretation:</b><br>
        The correlation heatmap shows how price relates to various factors numerically. 
        Positive correlations (red shades) suggest that higher values of a factor are associated 
        with higher prices, while negative correlations (blue shades) indicate an inverse relationship. 
        For example, strong correlations with item_group_enc or item_category_enc highlight key drivers 
        influencing price trends in Pasar Mini.
        </div>
        """,
        unsafe_allow_html=True
    )

# 2. Calculate Pearson correlation

# -----------------------------
# Pearson Correlation: Price vs Time
# -----------------------------

with st.container():
    st.subheader("🔗 Pearson Correlation: Price vs Time")
    st.caption(
        "This analysis examines the linear relationship between price and time."
        "to understand how prices change over the recorded dates in Pasar Mini."
    )

    # --- Convert date to datetime and then numeric ---
    pasar_mini_df['date_dt'] = pd.to_datetime(pasar_mini_df['date'], errors='coerce')
    pasar_mini_df['date_num'] = pasar_mini_df['date_dt'].map(pd.Timestamp.toordinal)

    # 1. Compute Pearson correlation
    pearson_corr = pasar_mini_df[['price', 'date_num']].corr(method='pearson')

    # 2. Heatmap
    fig = px.imshow(
        pearson_corr,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',  # red-blue diverging palette
        zmin=-1,
        zmax=1,
        title='Pearson Correlation: Price vs Time (Pasar Mini)'
    )

    fig.update_layout(
        title=dict(
            text='Pearson Correlation: Price vs Time (Pasar Mini)',
            x=0.5,
            xanchor='center'
        ),
        coloraxis_colorbar=dict(title="Correlation"),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Show correlation table
    st.dataframe(pearson_corr, use_container_width=True)

    # 4. Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF3E0;
            border-left:6px solid #FB8C00;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        After converting dates to numeric values, the Pearson correlation measures 
        the linear trend of prices over time. 
        A positive correlation indicates that prices increase as time progresses, 
        while a negative correlation suggests a decreasing trend. 
        This helps identify whether time is a key driver of price changes in Pasar Mini.
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("---")

# -----------------------------
# Regression Analysis
# -----------------------------

with st.container():
    st.subheader("📈 Multiple Linear Regression: Key Drivers of Price")

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

st.markdown("---")


# -----------------------------
# Statistical Testing 
# -----------------------------

# --- Display ANOVA result in Streamlit ---
st.subheader("📊 ANOVA: Price Differences Across Item Categories")
st.caption(" ANOVA test examines whether the average prices differ significantly "
    "between item categories in Pasar Mini."
) 
# Fit ANOVA model
anova_model = ols('price ~ C(item_category)', data=pasar_mini_df).fit()

fig = px.box(
        pasar_mini_df,
        x='item_category',
        y='price',
        color='item_category',
        color_discrete_sequence=px.colors.diverging.RdBu,
        title='Price Distribution by Item Category (Pasar Mini)',
        labels={
            'item_category': 'Item Category',
            'price': 'Price (RM)'
        }
    )


# Perform ANOVA
anova_result = sm.stats.anova_lm(anova_model, typ=2)


st.dataframe(anova_result, use_container_width=True)

st.markdown(
    """
    <div style="
        background-color:#FFF7ED;
        border-left:6px solid #FB923C;
        padding:16px;
        border-radius:10px;
        margin-top:12px;
    ">
    <b>Interpretation:</b><br>
    The price distributions differ noticeably across item categories, indicating varying pricing 
    structures within Pasar Mini. Categories with wider spreads and visible outliers suggest 
    greater price variability, possibly driven by differences in product types or demand levels. 
    These variations highlight item category as an important factor influencing price behavior.
    </div>
    """,
    unsafe_allow_html=True
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
# 1. Segmentation Analysis 
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
        filtered_seg[['state_enc', 'state', 'item_group_enc', 'item_group','price']].head(10),
        use_container_width=True
    )
# Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF3E0;
            border-left:6px solid #FB8C00;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        The table and chart show that certain states have consistently higher average prices 
        across specific item groups. This suggests that both regional location and product category 
        influence pricing trends in Pasar Mini. Retailers can use these insights to identify key 
        areas where price adjustments or promotional strategies may be most effective.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# 2. Segmentation Analysis
# -----------------------------
with st.container():
    st.subheader("📊 Average Price by Item Category")
    st.caption(
        "This analysis shows the average price and transaction count for each item category "
        "in Pasar Mini, helping identify high-demand and high-priced categories."
    )

    # 1. Aggregate data
    seg_category = (
        pasar_mini_df.groupby('item_category')['price']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('count', ascending=False)
    )

    # 2. Bar chart
    fig = px.bar(
        seg_category,
        x='item_category',
        y='mean',
        title='Average Price by Item Category (Pasar Mini-Categorical)',
        color_discrete_sequence=['darkblue'],
        labels={
            'item_category': 'Item Category',
            'mean': 'Average Price (RM)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Table in dropdown (expander)
    with st.expander("📄 View Item Category Table", expanded=False):
        st.dataframe(seg_category, use_container_width=True)

    # 4. Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF7ED;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        The bar chart shows that some item categories have higher average prices than others, 
        reflecting differences in product types and demand levels. Categories with more transactions 
        indicate popular items, while the average price highlights high-value categories. 
        Together, this helps identify key product categories influencing price trends in Pasar Mini.
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------
# Drill-Down Analysis 
# -----------------------------

st.subheader("🔎 Drill-Down Analysis: Top Items in Johor (Pasar Mini)")

st.caption(
    "This drill-down analysis focuses on Johor to identify the top-selling items."
    "and examine their contribution to price trends and transaction volumes in Pasar Mini."
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
        <b>Interpretation:</b><br>
        The results indicate that a small number of items dominate transaction volumes in Johor, 
        suggesting strong and consistent consumer demand for essential goods. 
        This concentration reflects localized consumption patterns influenced by daily necessities 
        rather than price variability alone. Such trends highlight how regional purchasing behavior
        acts as a key driver of observed price stability and frequency in Pasar Mini markets.   
       </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------
# Root Cause Analysis: Standardized Regression
# -----------------------------
with st.container():
    st.subheader("🧩 Root Cause Analysis: Standardized Regression")
    st.caption(
        "This analysis identifies key factors driving price variations in Pasar Mini "
        "using standardized regression coefficients."
    )

    # 1. Prepare features
    features = pasar_mini_df[['state_enc', 'district_enc', 'item_group_enc', 'item_category_enc']]
    feature_names = ['state_enc', 'district_enc', 'item_group_enc', 'item_category_enc']

    # 2. Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_scaled = sm.add_constant(X_scaled)  # add constant for regression

    # 3. Fit OLS regression
    rca_reg = sm.OLS(pasar_mini_df['price'], X_scaled).fit()

    # 4. Prepare coefficient dataframe
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': rca_reg.params[1:]  # skip constant
    }).sort_values(by='Impact', key=abs, ascending=False)

    # 5. Display table of impacts
    st.markdown("### 📄 Standardized Regression Coefficients")
    st.dataframe(coef_df, use_container_width=True)

    # 6. Plot bar chart
    fig = px.bar(
        coef_df,
        x='Feature',
        y='Impact',
        title='Root Cause Impact Strength (Standardized Regression)',
        color_discrete_sequence=['darkblue']
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")  # zero reference line
    st.plotly_chart(fig, use_container_width=True)

    # 7. Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF7ED;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        The standardized regression highlights which factors have the strongest impact on price. 
        Positive coefficients indicate that higher values of a feature increase the price, 
        while negative coefficients decrease it. In this dataset, the analysis shows that certain 
        features, such as item category or state, are key drivers behind price variations in Pasar Mini.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Pareto Analysis: Top Items Driving Price
# -----------------------------
with st.container():
    st.subheader("📊 Pareto Analysis: Top Items Driving Price (Pasar Mini)")
    st.caption(
        "This analysis identifies which items contribute the most to the overall price, "
        "allowing a focus on the few items driving the majority of revenue."
    )

    # 1. Prepare Pareto dataframe
    pareto_df = (
        pasar_mini_df
        .groupby('item')['price']
        .mean()
        .reset_index()
        .sort_values('price', ascending=False)
    )
    pareto_df['cum_pct'] = pareto_df['price'].cumsum() / pareto_df['price'].sum()

    # 2. Plot Pareto chart
    fig = go.Figure()

    # Bar for individual item contribution
    fig.add_trace(go.Bar(
        x=pareto_df.head(15)['item'],
        y=pareto_df.head(15)['price'],
        name='Average Price',
        marker_color='darkblue'
    ))

    # Line for cumulative percentage
    fig.add_trace(go.Scatter(
        x=pareto_df.head(15)['item'],
        y=pareto_df.head(15)['cum_pct'],
        mode='lines+markers',
        name='Cumulative %',
        marker=dict(color='orange'),
        yaxis='y2'
    ))

    # Create secondary y-axis for cumulative %
    fig.update_layout(
        title='Pareto Analysis: Top Items Driving Price (Pasar Mini)',
        xaxis_title='Item',
        yaxis_title='Average Price',
        yaxis2=dict(
            title='Cumulative %',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Dropdown table for top items
    with st.expander("📋 View Top 10 Items (Dropdown)"):
        selected_n = st.slider("Select number of top items", min_value=5, max_value=15, value=10)
        st.dataframe(
            pareto_df.head(selected_n)[['item', 'price', 'cum_pct']],
            use_container_width=True
        )

    # 4. Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF4E5;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        The Pareto chart shows that a small number of items contribute the majority of the total price. 
        The cumulative percentage line highlights which items are most influential. 
        This helps prioritize monitoring and pricing strategies for items that drive the largest revenue in Pasar Mini.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Outlier Detection
# -----------------------------
with st.container():
    st.subheader("🔎 Outlier Detection in Pasar Mini Prices")
    st.caption(
        "This analysis identifies unusually high prices using the IQR method and highlights extreme outliers."
    )

    # 1. Compute IQR and identify outliers
    Q1 = pasar_mini_df['price'].quantile(0.25)
    Q3 = pasar_mini_df['price'].quantile(0.75)
    IQR = Q3 - Q1

    outliers = pasar_mini_df[pasar_mini_df['price'] > Q3 + 1.5 * IQR]

    # 2. Box plot
    fig = px.box(
        pasar_mini_df,
        y='price',
        title='Outlier Detection in Pasar Mini Prices',
        color_discrete_sequence=['darkblue']
    )

    # 3. Overlay extreme outliers
    fig.add_scatter(
        y=outliers['price'],
        x=[0]*len(outliers),  # align with box
        mode='markers',
        marker=dict(
            color='red',
            size=8,
            symbol='x'
        ),
        name='Extreme Outliers'
    )

    fig.update_layout(
        height=600,
        width=800,
        margin=dict(l=40, r=40, t=80, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 4. Dropdown table for top outliers
    with st.expander("📋 View Top Outliers (Dropdown)"):
        top_n = st.slider("Select number of top outliers to display", min_value=5, max_value=20, value=10)
        st.dataframe(
            outliers[['state', 'item_category', 'item', 'price']].head(top_n),
            use_container_width=True
        )

    # 5. Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF4E5;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        The box plot identifies the overall price distribution, with extreme prices highlighted in red. 
        These outliers represent unusually high prices in Pasar Mini markets and may indicate exceptional demand, 
        limited supply, or pricing errors. Monitoring these outliers helps understand unusual pricing trends and informs better decision-making.
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Root Cause Drill-Down
# -----------------------------
with st.container():
    st.subheader("🌞 Root Cause Drill-Down by State (Pasar Mini)")
    st.caption(
        "This analysis explores the key drivers behind pricing by examining the hierarchy of state, district, premise, and item."
    )

    # 1. Aggregate data for drill-down
    rca_chain = (
        pasar_mini_df
        .groupby(['premise', 'state', 'district', 'item'])['price']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('count', ascending=False)
    )

    # 2. Sunburst chart
    fig = px.sunburst(
        rca_chain.head(50),
        path=['state', 'district', 'premise', 'item'],
        values='count',
        color='state',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title='Root Cause Drill-Down by State in Pasar Mini'
    )
    st.plotly_chart(fig, use_container_width=True)

    # 3. Dropdown table for top 10
    with st.expander("📋 View Top 10 Root Cause Records (Dropdown)"):
        top_n = st.slider("Select number of top records to view", min_value=5, max_value=20, value=10)
        st.dataframe(
            rca_chain.head(top_n)[['state', 'district', 'premise', 'item', 'mean', 'count']],
            use_container_width=True
        )

    # 4. Interpretation
    st.markdown(
        """
        <div style="
            background-color:#FFF4E5;
            border-left:6px solid #FB923C;
            padding:16px;
            border-radius:10px;
            margin-top:12px;
        ">
        <b>Interpretation:</b><br>
        The sunburst chart reveals how pricing contributions cascade from state to district, premise, and item. 
        States with the largest counts dominate the distribution, highlighting which regions and premises drive most transactions. 
        This hierarchical insight helps identify critical areas to investigate further for price variations in Pasar Mini markets.
        </div>
        """,
        unsafe_allow_html=True
    )
