import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --------------------
# 1. Load Data
# --------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/refs/heads/main/dataset/pasar_mini_data_updated.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    pasar_mini_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()
    
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
    /* Unified Expander Style */
    .stExpander {
        background-color: #1E1E1E !important;
        border: 2px solid #4facfe !important;
        border-radius: 15px !important;
        animation: glow 3s infinite;
    }
    @keyframes glow {
        0% { box-shadow: 0 0 5px #4facfe; }
        50% { box-shadow: 0 0 20px #00f2fe; }
        100% { box-shadow: 0 0 5px #4facfe; }
    }
    .metric-card {
        background: #ffffff; border-radius: 12px; padding: 15px;
        text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-bottom: 4px solid #ddd; transition: 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
</style>
""", unsafe_allow_html=True)

# --------------------
# 3. Header Section
# --------------------
st.markdown('<div class="center-title">Descriptive Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Nurul Ain Maisarah Binti Hamidin | S22A0064</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------
# 4. Dataset Preview (CLOSED BY DEFAULT)
# --------------------
with st.expander("DATASET PREVIEW", expanded=False):
    st.write("### Previewing First 5 Rows")
    st.dataframe(pasar_mini_df.head(), use_container_width=True)

# --------------------
# 5. KPI Metrics
# --------------------
# --------------------
# KPI Metrics Styling
# --------------------
st.markdown("""
<style>
/* Card Styling */
.metric-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    border-top: 5px solid #ddd; 
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    min-height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

/* Individual accent colors using your palette */
.m-max { border-top-color: #FF4B4B; background: linear-gradient(to bottom, #fff5f5, #ffffff); } 
.m-min { border-top-color: #00CC96; background: linear-gradient(to bottom, #f0fff4, #ffffff); } 
.m-top { border-top-color: #636EFA; background: linear-gradient(to bottom, #f0f3ff, #ffffff); } 
.m-cat { border-top-color: #AB63FA; background: linear-gradient(to bottom, #f9f0ff, #ffffff); }

.metric-label {
    font-size: 0.75rem;
    color: #555;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #1f1f1f;
    margin: 2px 0;
}

.metric-help {
    font-size: 0.7rem;
    color: #888;
    line-height: 1.2;
    margin-top: 8px;
    border-top: 1px solid #eee;
    padding-top: 8px;
}
</style>
""", unsafe_allow_html=True)

st.subheader("Key Dataset Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card m-max">
        <div class="metric-label">🔺 Max Price</div>
        <div class="metric-value">RM 498.00</div>
        <div class="metric-help">Bawang Besar Import (India) 1 kg<br><b>2025-12-19</b></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card m-min">
        <div class="metric-label">🔻 Min Price</div>
        <div class="metric-value">RM 0.50</div>
        <div class="metric-help">Serbuk Kari Adabi<br><b>2025-12-08</b></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card m-top">
        <div class="metric-label">🏢 Top Premise</div>
        <div class="metric-value">1,641</div>
        <div class="metric-help">Kifarah Fresh Mart<br>Records Count</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card m-cat">
        <div class="metric-label">📦 Top Item Group</div>
        <div class="metric-value">67,098</div>
        <div class="metric-help">Barangan Berbungkus<br>Total Items</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# --------------------
# 6. Main Objective
# --------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Descriptive Objectives")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; color: white;">
    To descriptively analyze price patterns, distribution characteristics across time, location, and item classifications among Pasar Mini.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Visual Analysis")

# --------------------
# 7. Visualisation: Average Price Over Time (CLOSED BY DEFAULT)
# --------------------
avg_price = pasar_mini_df.groupby('date')['price'].mean().reset_index()

st.markdown("""
<div style="background: linear-gradient(90deg, #FF4081 0%, #764ba2 100%); padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To examine trends and changes in average item prices over time.
</div>
""", unsafe_allow_html=True)

with st.expander(" AVERAGE PRICE OVER TIME ANALYSIS", expanded=False):
    # --- Line Chart ---
    fig_line = px.line(avg_price, x='date', y='price', markers=True,
                  labels={'date': 'Date', 'price': 'Average Price (RM)'},
                  line_shape='spline', color_discrete_sequence=['#FF4081'])
    
    fig_line.update_layout(
        hovermode='x unified', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        font=dict(color="#FF4081"),
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- Trend Summary Points ---
    st.markdown("### Temporal Trend Insights")
    
    st.info("""
    * **Overall Movement:** December 2025 shows a **fluctuating yet downward trend**, starting at **RM 12.29** (Dec 1st) and ending at **RM 11.25** (Dec 31st).
    * **Weekly Peaks:** Data reveals a **cyclical pattern** with peaks every 7 days (1st, 8th, 15th, and 22nd). The monthly high was **RM 12.61** on December 22nd.
    * **Rapid Adjustments:** Following each peak, prices drop abruptly by ~1.0 unit. A clear example is the shift from **RM 12.29** (Dec 1st) to **RM 11.08** (Dec 2nd).
    * **Volatility Range:** Most prices oscillate within a narrow band of **RM 11.50 to RM 12.00**, with the lowest point reached on December 26th (**RM 10.94**).
    * **Temporal Rhythm:** The consistent 7-day surge cycle identifies a recurring reporting rhythm or specific weekly adjustment behavior in the Pasar Mini structure.
    """)

    # --- Data Table ---
    st.markdown("#### Daily Price Summary (First 10 Days)")
    st.dataframe(avg_price.head(10), use_container_width=True)

# --------------------
# Central Tendency 
# --------------------
# --------------------
# 8. Visualisation: Central Tendency 
# --------------------
price_mean = pasar_mini_df['price'].mean()
price_median = pasar_mini_df['price'].median()
price_mode = pasar_mini_df['price'].mode()[0] if not pasar_mini_df['price'].mode().empty else 0
price_count = len(pasar_mini_df)

st.markdown("""
<div style="background: linear-gradient(90deg, #764ba2 0%, #4facfe 100%); padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>🎯 Objective:</strong> To identify typical price levels (Mean, Median, Mode) within the dataset.
</div>
""", unsafe_allow_html=True)

# Expander set to False to start closed
with st.expander("📊 Measures of Central Tendency for price", expanded=False):
    
    # --- Chart Section ---
    measures = ['Mean', 'Median', 'Mode']
    values = [price_mean, price_median, price_mode]
    
    fig_bar = go.Figure(data=[go.Bar(
        x=measures, y=values,
        marker_color=['#4facfe', '#764ba2', '#00f2fe'],
        text=[f'RM {v:.2f}' for v in values], textposition='auto'
    )])
    
    fig_bar.update_layout(
        title_text="Central Tendency Analysis", 
        title_x=0.5, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- Point Summary Section ---
    st.markdown("### 📝 Statistical Summary")
    
    st.info(f"""
    * **Data Volume:** The descriptive analysis reveals a significant volume of market data, with a total count of **{price_count:,}** recorded price points.
    * **Mean:** The average item price is calculated at **RM {price_mean:.2f}**.
    * **Median:** The middle value of the price distribution stands at **RM {price_median:.2f}**.
    * **Mode:** The most frequently occurring price point in the dataset is **RM {price_mode:.2f}**.
    """)

    # --- Data Table ---
    st.markdown("#### 📋 Detailed Metrics")
    ct_df = pd.DataFrame({
        'Measure': ['Total Count', 'Mean', 'Median', 'Mode'],
        'Value': [f"{price_count:,}", f"RM {price_mean:.2f}", f"RM {price_median:.2f}", f"RM {price_mode:.2f}"]
    })
    st.table(ct_df)
# --------------------
# Measures of Dispersion
# --------------------

# Calculations for Dispersion
price_std = pasar_mini_df['price'].std()
price_min = pasar_mini_df['price'].min()
price_max = pasar_mini_df['price'].max()
price_range = price_max - price_min
q1 = pasar_mini_df['price'].quantile(0.25)
q3 = pasar_mini_df['price'].quantile(0.75)
iqr = q3 - q1

# Create a summary DataFrame for the Dispersion Metrics
price_stats_df = pd.DataFrame({
    'Measure': ['Std Deviation', 'Range', '25th Percentile (Q1)', '75th Percentile (Q3)', 'IQR'],
    'Value': [price_std, price_range, q1, q3, iqr]
})

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #1f77b4 0%, #2ca02c 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To summarise the variability and range of item prices to understand the extent of price dispersion.
</div>
""", unsafe_allow_html=True)

# Expander with Prominent Icon
with st.expander("Statistical Measures for Price Dispersion", expanded=False):
    
    # Create Interactive Bar Chart
    fig_disp = px.bar(
        price_stats_df,
        x='Measure',
        y='Value',
        color='Measure',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        title="Dispersion Analysis (Spread of Prices)",
        text='Value'
    )

    fig_disp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_disp.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#7f7f7f"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20)
    )

    st.plotly_chart(fig_disp, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Dispersion Insights")
    st.info(f"""
    * **Price Volatility:** To understand price volatility, measures of dispersion were employed. The **Standard Deviation of {price_std:.2f}** and a **Range of {price_range:.2f}** highlight a wide variety in the pricing of consumer goods.
    * **Market Diversity:** This wide range is likely due to the inclusion of both small units (e.g., timun) and bulk or premium items reaching a maximum of **RM {price_max:.2f}**.
    * **Stable Baseline:** The **Interquartile Range (IQR) of {iqr:.2f}** provides a more stable baseline, showing that the middle 50% of all items fall between **RM {q1:.2f} (25th percentile)** and **RM {q3:.2f} (75th percentile)**.
    * **Typical Spending:** This middle range represents the typical expenditure for a standard consumer unit in mini-markets.
    """)

    # Data Table
    st.markdown("#### Dispersion Metric Details")
    st.table(price_stats_df)
# --------------------
# 10. Visualisation: Cumulative Frequency Analysis
# --------------------

# Data Processing for Cumulative Plot
price_data = pasar_mini_df['price'].sort_values().reset_index(drop=True)
total_count = len(price_data)
cumulative_counts = price_data.value_counts(sort=False).sort_index().cumsum()
cumulative_percentages = (cumulative_counts / total_count) * 100

cumulative_df = pd.DataFrame({
    'price': cumulative_percentages.index,
    'cumulative_percentage': cumulative_percentages.values
})

# Recalculate key stats for annotations
p_min, p_max = price_data.min(), price_data.max()
p_med = pasar_mini_df['price'].median()
p_q1 = pasar_mini_df['price'].quantile(0.25)
p_q3 = pasar_mini_df['price'].quantile(0.75)

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #FF69B4 0%, #764ba2 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To provide a comprehensive statistical overview of item prices using descriptive summary statistics.
</div>
""", unsafe_allow_html=True)

# Expander with Prominent Icon
with st.expander(" Detailed Summary Statistics & Cumulative Analysis", expanded=False):
    
    # Create the Plotly figure
    fig_cum = go.Figure()

    # Add the cumulative frequency line
    fig_cum.add_trace(go.Scatter(
        x=cumulative_df['price'], y=cumulative_df['cumulative_percentage'],
        mode='lines', name='Cumulative %', line=dict(color='#FF69B4', width=3)
    ))

    # Add horizontal/vertical indicator lines for Median
    fig_cum.add_shape(type="line", x0=p_min, y0=50, x1=p_med, y1=50, line=dict(color="white", width=1, dash="dash"))
    fig_cum.add_shape(type="line", x0=p_med, y0=0, x1=p_med, y1=50, line=dict(color="white", width=1, dash="dash"))

    fig_cum.update_layout(
        title_text='Cumulative Price Distribution',
        xaxis_title='Price (RM)',
        yaxis_title='Cumulative Percentage (%)',
        hovermode='x unified',
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#FF69B4"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )

    # Annotations for key points
    fig_cum.add_annotation(x=p_med, y=50, text=f"Median: RM{p_med:.2f}", showarrow=True, arrowhead=1, ax=-40, ay=-40)
    fig_cum.add_annotation(x=p_q1, y=25, text=f"Q1: RM{p_q1:.2f}", showarrow=True, arrowhead=1)
    fig_cum.add_annotation(x=p_q3, y=75, text=f"Q3: RM{p_q3:.2f}", showarrow=True, arrowhead=1)

    st.plotly_chart(fig_cum, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Cumulative Analysis Insights")
    
    st.info(f"""
    * **Distribution Concentration:** The analysis reveals a distribution heavily concentrated at the lower end, confirming a **significant positive skew** (2.298).
    * **Median Threshold:** The rapid accumulation aligns with the **Median price of {p_med:.2f}**, indicating that 50% of all items are priced at or below this RM 9.00 threshold.
    * **Market Character:** The data demonstrates that lower price points possess higher individual counts. This suggests the "Pasar Mini" sector primarily services **high-volume, low-cost essential goods**.
    * **Extreme Outliers:** While the scale reaches RM {p_max:.2f}, these higher points appear with a frequency of only one, characterizing them as extreme outliers rather than representative market trends.
    """)

    # --- Summary Statistics Table ---
    st.markdown("#### Summary Statistics Table")
    st.dataframe(pasar_mini_df['price'].describe().to_frame().T, use_container_width=True)
# --------------------
# 11. Visualisation: Measures of Distribution Shape
# --------------------

# Calculate measures
price_skewness = pasar_mini_df['price'].skew()
price_kurtosis = pasar_mini_df['price'].kurt()

# Create a DataFrame for display
distribution_shape_df = pd.DataFrame({
    'Measure': ['Skewness', 'Kurtosis'],
    'Value': [price_skewness, price_kurtosis]
})

# Section Objective Header (Using a distinct gradient)
st.markdown("""
<div style="background: linear-gradient(90deg, #d62728 0%, #764ba2 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To assess the shape of the price distribution in order to identify skewness and concentration of prices.
</div>
""", unsafe_allow_html=True)

# Expander with Prominent Icon (Closed by default)
with st.expander("Measures of Distribution Shape for Price", expanded=False):
    
    # Create the interactive bar chart
    fig_shape = px.bar(
        distribution_shape_df,
        x='Measure',
        y='Value',
        color='Measure',
        color_discrete_sequence=['#ff7f0e', '#d62728'], # High contrast colors
        title="Distribution Shape (Skewness & Kurtosis)",
        text='Value'
    )

    fig_shape.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_shape.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#7f7f7f"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20)
    )

    st.plotly_chart(fig_shape, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Distribution Shape Insights")
    st.info(f"""
    * **Right-Skewed Distribution:** A **Skewness of {price_skewness:.2f}** indicates a significant right-skewed pattern. This confirms that while most goods are priced below **RM 10.00**, a few high-priced items pull the average upward.
    * **Heavy-Tailed Extremes:** A high **Kurtosis of {price_kurtosis:.2f}** suggests a "heavy-tailed" distribution. This statistically confirms the presence of extreme price outliers within the dataset.
    * **Outlier Impact:** The most prominent outlier identified is the maximum recorded value of **RM 498.00**, which significantly influences the tail of the distribution compared to the typical item price.
    """)

    # Data Table
    st.markdown("#### Detailed Shape Metrics")
    st.table(distribution_shape_df.style.format({"Value": "{:.3f}"}))

# --------------------
# 12. Visualisation: Mode for Categorical Columns
# --------------------

# Calculations (Recalculating to ensure values are current)
# Replace these with your actual categorical columns if names differ
cat_cols = ['item_group', 'item_category', 'item', 'unit', 'state', 'district', 'premise']
mode_data = []

for col in cat_cols:
    if col in pasar_mini_df.columns:
        mode_val = pasar_mini_df[col].mode()[0]
        mode_data.append({'Column': col, 'Mode': mode_val})

selected_modes_df = pd.DataFrame(mode_data)

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #00CC96 0%, #636EFA 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To identify the most frequently occurring categories within selected categorical variables.
</div>
""", unsafe_allow_html=True)

# Expander with Prominent Icon (Closed by default)
with st.expander("Mode Categorical Columns", expanded=False):
    
    # Create the interactive bar chart
    fig_cat = px.bar(
        selected_modes_df,
        x='Column',
        y='Mode',
        color='Column',
        title='Most Frequent Categorical Values',
        text='Mode',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_cat.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#7f7f7f"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50, b=20),
        height=500
    )
    
    fig_cat.update_traces(textposition='outside')
    st.plotly_chart(fig_cat, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Categorical Analysis Insights")
    st.info("""
    * **Dominant Product Groupings:** Analysis reveals that **"Barangan Berbungkus"** (Packaged Goods) is the most frequent item_group, while **"Sayur-Sayuran"** (Vegetables) is the modal item_category.
    * **Leading Consumer Items:** The most frequently recorded individual product is **"Minyak Masak Tulen Cap Buruh (1kg)"**, suggesting essential cooking oils are a primary focus of price monitoring.
    * **Geographical Baseline:** The dataset is most heavily represented by the state of **Johor** and the district of **Seberang Perai Utara**.
    * **Primary Representative:** **"Pasar Raya Kifarah Fresh Mart"** was identified as the modal premise, serving as the primary representative for the mini-market category in this analysis.
    """)

    # Data Table
    st.markdown("#### Modal Value Details")
    st.table(selected_modes_df)

# --------------------
# 13. Visualisation: Cumulative Frequency for Dates
# --------------------

# Data Processing
pasar_mini_df['date'] = pd.to_datetime(pasar_mini_df['date'])

# Filter for Dec 2025 (up to the 22nd)
filtered_df_date = pasar_mini_df[
    (pasar_mini_df['date'].dt.year == 2025) & 
    (pasar_mini_df['date'].dt.month == 12) & 
    (pasar_mini_df['date'].dt.day <= 22)
].copy()

# Aggregations
date_counts = filtered_df_date['date'].value_counts().sort_index().reset_index()
date_counts.columns = ['date', 'count']
average_price_per_date = filtered_df_date.groupby('date')['price'].mean().reset_index()
date_counts = date_counts.merge(average_price_per_date, on='date', how='left')

# Cumulative calculations
date_counts['cumulative_count'] = date_counts['count'].cumsum()
date_counts['cumulative_percentage'] = (date_counts['cumulative_count'] / len(filtered_df_date)) * 100

# Highlighting specific surge dates
specific_dates = [
    pd.to_datetime('2025-12-01'), pd.to_datetime('2025-12-08'), 
    pd.to_datetime('2025-12-15'), pd.to_datetime('2025-12-22')
]

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #2ECC71 0%, #27AE60 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To examine the accumulation of price observations over time using cumulative frequency analysis.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander(" Cumulative Frequency for Dates", expanded=False):
    
    # Create Chart
    fig_date = px.line(
        date_counts, x='date', y='cumulative_count', markers=True,
        title='Data Accumulation Surge (Dec 1st - 22nd)',
        labels={'date': 'Date', 'cumulative_count': 'Total Entries Accumulation'},
        color_discrete_sequence=['#2ECC71']
    )

    # Add Star markers for surge dates
    for s_date in specific_dates:
        if s_date in date_counts['date'].values:
            row = date_counts[date_counts['date'] == s_date]
            fig_date.add_scatter(
                x=[s_date], y=[row['cumulative_count'].iloc[0]],
                mode='markers', marker=dict(size=12, color='red', symbol='star'),
                name=f"Surge: {s_date.strftime('%b %d')}",
                hovertext=f"Surge Date: {s_date.strftime('%Y-%m-%d')}<br>Cumulative %: {row['cumulative_percentage'].iloc[0]:.2f}%"
            )

    fig_date.update_layout(
        hovermode='x unified', title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#2ECC71"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_date, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Temporal Accumulation Insights")
    st.info("""
    * **Non-Uniform Collection:** The analysis reveals a highly skewed data collection pattern, heavily concentrated on four specific dates: **Dec 1st, 8th, 15th, and 22nd**.
    * **Dominant Surge Dates:** These four dates alone account for approximately **87.78%** of the entire dataset, with December 22nd contributing the highest volume (36,378 records).
    * **Weekly Cycles:** This periodic spike suggests a **synchronized data harvest** or weekly reporting cycle from the PriceCatcher platform.
    * **Representation Note:** The dataset offers high granularity for specific snapshots, reaching **97.14%** accumulation by Dec 22nd, reflecting weekly cycles rather than a smooth daily distribution.
    """)

    # Data Table for verification
    st.markdown("#### Cumulative Data Log")
    st.dataframe(date_counts.set_index('date'), use_container_width=True)

# --------------------
# 14. Visualisation: Cumulative Frequency for Item
# --------------------

# Data Processing
item_counts = pasar_mini_df['item'].value_counts().reset_index()
item_counts.columns = ['item', 'count']

# Calculate average price per item for richer hover data
average_price_per_item = pasar_mini_df.groupby('item')['price'].mean().reset_index()
average_price_per_item.rename(columns={'price': 'average_price'}, inplace=True)
item_counts = item_counts.merge(average_price_per_item, on='item', how='left')

# Sort and calculate cumulative statistics
item_counts = item_counts.sort_values(by='count', ascending=False)
total_rows = len(pasar_mini_df)
item_counts['percentage'] = (item_counts['count'] / total_rows) * 100
item_counts['cumulative_count'] = item_counts['count'].cumsum()
item_counts['cumulative_percentage'] = item_counts['percentage'].cumsum()

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #4CAF50 0%, #2E7D32 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To identify dominant items contributing to the majority of observations through cumulative frequency visualisation.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander(" Cumulative Frequency Visualization for Item", expanded=False):
    
    # Create the Top 15 Bar Chart
    fig_item = px.bar(
        item_counts.head(15), 
        x='item', y='count', color='item',
        title='Top 15 Most Frequent Items',
        labels={'item': 'Product Name', 'count': 'Frequency'},
        hover_data={
            'percentage': ':.2f%',
            'cumulative_percentage': ':.2f%',
            'average_price': 'RM {:.2f}'
        },
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    fig_item.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#4CAF50"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        height=600
    )

    st.plotly_chart(fig_item, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Market Concentration Insights")
    st.info("""
    * **Concentrated Structure:** Out of 252 unique items, the dataset is dominated by a small number of high-frequency essential goods, specifically cooking oil brands.
    * **Top Prevalent Items:** **'Minyak masak tulen cap buruh'** is the most frequent (2.12%), followed by **'Cap Seri Murni'** (2.08%) and **'Cap Pisau'** (1.93%).
    * **The Top 5 Impact:** Collectively, the top five items account for **9.36%** of all recorded observations, showing a significant Pareto-type skew.
    * **The Long-Tail Distribution:** Conversely, a large variety of products like specialized rice or niche perishables appear very rarely. This suggests that mini-market data is heavily weighted toward a narrow selection of high-turnover household staples.
    """)

    # Data Table
    st.markdown("#### Item Frequency Rank (Top 15)")
    st.dataframe(
        item_counts.head(15)[['item', 'count', 'percentage', 'cumulative_percentage', 'average_price']],
        use_container_width=True
    )

# --------------------
# 15. Visualisation: Cumulative Frequency for Item Group
# --------------------

# Data Processing
item_group_counts = pasar_mini_df['item_group'].value_counts().reset_index()
item_group_counts.columns = ['item_group', 'count']

# Calculate average price per item_group
average_price_per_item_group = pasar_mini_df.groupby('item_group')['price'].mean().reset_index()
average_price_per_item_group.rename(columns={'price': 'average_price'}, inplace=True)
item_group_counts = item_group_counts.merge(average_price_per_item_group, on='item_group', how='left')

# Calculate proportions and cumulative metrics
total_rows = len(pasar_mini_df)
item_group_counts['percentage'] = (item_group_counts['count'] / total_rows) * 100
item_group_counts['cumulative_percentage'] = item_group_counts['percentage'].cumsum()

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #4CAF50 0%, #00f2fe 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To analyse the contribution of different item groups to the overall dataset using cumulative frequency analysis.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander("Cumulative Frequency Visualization for Item Group", expanded=False):
    
    # Create the Interactive Bar Chart
    fig_group = px.bar(
        item_group_counts, 
        x='item_group', y='count', color='item_group',
        title='Item Group Frequency Distribution',
        labels={'item_group': 'Item Group', 'count': 'Number of Entries'},
        hover_data={
            'percentage': ':.2f%',
            'cumulative_percentage': ':.2f%',
            'average_price': 'RM {:.2f}'
        },
        color_discrete_sequence=px.colors.qualitative.Dark2
    )

    fig_group.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#4CAF50"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )

    st.plotly_chart(fig_group, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Item Group Insights")
    st.info("""
    * **Category Dominance:** 'Barangan berbungkus' (packaged goods) is the dominant category with **43.96%** of total observations.
    * **Major Pairs:** Together with 'Barangan segar' (fresh produce), these two groups represent a massive **78.48%** of all records.
    * **Core Retail Baseline:** Coverage expands to **98.03%** when dry goods and baby products are included, defining the primary focus of the Pasar Mini inventory.
    * **Marginal Proportions:** Beverages and cleaning products account for only **1.51%** and **0.45%** respectively, remaining statistically secondary in the observation volume.
    """)

    # Data Table
    st.markdown("#### Item Group Statistical Breakdown")
    st.dataframe(
        item_group_counts[['item_group', 'count', 'percentage', 'cumulative_percentage', 'average_price']],
        use_container_width=True
    )

# --------------------
# 16. Visualisation: Cumulative Frequency for Item Category
# --------------------

# Data Processing
item_cat_counts = pasar_mini_df['item_category'].value_counts().reset_index()
item_cat_counts.columns = ['item_category', 'count']

# Calculate average price per category for hover data
avg_price_cat = pasar_mini_df.groupby('item_category')['price'].mean().reset_index()
avg_price_cat.rename(columns={'price': 'average_price'}, inplace=True)
item_cat_counts = item_cat_counts.merge(avg_price_cat, on='item_category', how='left')

# Sort and calculate cumulative percentages
item_cat_counts = item_cat_counts.sort_values(by='count', ascending=False)
total_rows = len(pasar_mini_df)
item_cat_counts['percentage'] = (item_cat_counts['count'] / total_rows) * 100
item_cat_counts['cumulative_percentage'] = item_cat_counts['percentage'].cumsum()

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #4CAF50 0%, #1b5e20 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To evaluate the distribution and dominance of item categories based on cumulative frequency patterns.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander("Cumulative Frequency Visualization for item_category", expanded=False):
    
    # Create the Interactive Bar Chart
    fig_cat = px.bar(
        item_cat_counts, 
        x='item_category', y='count', color='item_category',
        title='Distribution of Observations by Item Category',
        labels={'item_category': 'Category', 'count': 'Number of Entries'},
        hover_data={
            'percentage': ':.2f%',
            'cumulative_percentage': ':.2f%',
            'average_price': 'RM {:.2f}'
        },
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig_cat.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#4CAF50"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig_cat, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Item Category Insights")
    st.info("""
    * **Category Dominance:** 'Sayur-sayuran' (vegetables) is the primary driver of the dataset, with **33,018 entries (21.63%)**.
    * **Core Concentration:** The top two categories (Vegetables + Oils/Fats) represent over **32%** of all observations. 
    * **Major Drivers:** The top five categories (including packaged spices, baby milk, and sauces) constitute **55.55%** of the data, while the top ten account for nearly **74.74%**.
    * **Long-Tail Effect:** A significant "long-tail" exists where niche categories like 'mentega' contribute as little as **0.01%**, underscoring a market baseline focused almost entirely on daily food essentials.
    """)

    # Data Table (Top 10)
    st.markdown("#### Top 10 Category Statistical Breakdown")
    st.dataframe(
        item_cat_counts.head(10)[['item_category', 'count', 'percentage', 'cumulative_percentage', 'average_price']],
        use_container_width=True
    )

# --------------------
# 17. Visualisation: Cumulative Frequency for Premise
# --------------------

# Data Processing: Focus on top 15 premises
premise_counts = pasar_mini_df['premise'].value_counts().reset_index().head(15)
premise_counts.columns = ['premise', 'count']

# Calculate average price per premise for hover context
avg_price_premise = pasar_mini_df.groupby('premise')['price'].mean().reset_index()
avg_price_premise.rename(columns={'price': 'average_price'}, inplace=True)
premise_counts = premise_counts.merge(avg_price_premise, on='premise', how='left')

# Calculate proportions and cumulative metrics (based on entire dataset for accuracy)
total_global_rows = len(pasar_mini_df)
premise_counts['percentage'] = (premise_counts['count'] / total_global_rows) * 100
premise_counts['cumulative_percentage'] = premise_counts['percentage'].cumsum()

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #1F77B4 0%, #084B82 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To determine the concentration of observations among the top premises contributing to the dataset.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander("Cumulative Frequency Visualization for Premise", expanded=False):
    
    # Create the Interactive Bar Chart
    fig_premise = px.bar(
        premise_counts, 
        x='premise', y='count', color='premise',
        title='Top 15 Premises by Observation Volume',
        labels={'premise': 'Retailer Name', 'count': 'Number of Entries'},
        hover_data={
            'percentage': ':.2f%',
            'cumulative_percentage': ':.2f%',
            'average_price': 'RM {:.2f}'
        },
        color_discrete_sequence=px.colors.sequential.Blues_r
    )

    fig_premise.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#1F77B4"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig_premise, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Premise Distribution Insights")
    st.info("""
    * **Decentralized Network:** Unlike product categories, premises show a balanced distribution. The top contributor, **PASAR RAYA KIFARAH FRESH MART**, accounts for only **1.08%** of entries.
    * **Low Concentration:** The top five premises collectively represent only **5.08%** of total observations, ensuring the dataset is not skewed by a single dominant retailer.
    * **Broad Representation:** It takes approximately 40 to 50 premises to reach a 50% cumulative threshold, reflecting a wide geographical reporting network.
    * **Balanced Baseline:** The distribution confirms that the findings are representative of a balanced network of multiple small-scale commercial entities rather than localized anomalies.
    """)

    # Data Table
    st.markdown("#### Top 15 Premise Statistical Breakdown")
    st.dataframe(
        premise_counts[['premise', 'count', 'percentage', 'cumulative_percentage', 'average_price']],
        use_container_width=True
    )

# --------------------
# 18. Visualisation: Cumulative Frequency for State
# --------------------

# Data Processing
state_counts = pasar_mini_df['state'].value_counts().reset_index()
state_counts.columns = ['state', 'count']

# Calculate average price per state for hover context
avg_price_state = pasar_mini_df.groupby('state')['price'].mean().reset_index()
avg_price_state.rename(columns={'price': 'average_price'}, inplace=True)
state_counts = state_counts.merge(avg_price_state, on='state', how='left')

# Sort and calculate cumulative metrics
state_counts = state_counts.sort_values(by='count', ascending=False)
total_rows = len(pasar_mini_df)
state_counts['percentage'] = (state_counts['count'] / total_rows) * 100
state_counts['cumulative_percentage'] = state_counts['percentage'].cumsum()

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #8B008B 0%, #4B0082 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To assess the distribution of price observations across states using cumulative frequency analysis.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander("Cumulative Frequency Visualization for State", expanded=False):
    
    # Create the Interactive Bar Chart
    fig_state = px.bar(
        state_counts, 
        x='state', y='count', color='state',
        title='Observation Volume by State',
        labels={'state': 'State', 'count': 'Number of Entries'},
        hover_data={
            'percentage': ':.2f%',
            'cumulative_percentage': ':.2f%',
            'average_price': 'RM {:.2f}'
        },
        color_discrete_sequence=px.colors.qualitative.Vivid
    )

    fig_state.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#8B008B"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig_state, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Geographical Distribution Insights")
    st.info("""
    * **Regional Concentration:** The dataset shows moderate concentration, with **Johor** being the primary contributor at **12.41%** (18,944 entries).
    * **The Top 5 Milestone:** Johor, Perak, Sarawak, Pulau Pinang, and Kelantan collectively account for **50.25%** of all observations. 
    * **Reporting Bias:** Half of the national monitoring data comes from only one-third of the administrative regions, suggesting more intensive activity in larger or more populous states.
    * **Lower Representation:** Regions like W.P. Labuan (0.38%) and Perlis (1.64%) contribute significantly less, implying the baseline findings are heavily influenced by the top 5 states.
    """)

    # Data Table (Top 10)
    st.markdown("#### Top 10 State Statistical Breakdown")
    st.dataframe(
        state_counts.head(10)[['state', 'count', 'percentage', 'cumulative_percentage', 'average_price']],
        use_container_width=True
    )

# --------------------
# 19. Visualisation: Cross-tabulation Heatmap (Price & State)
# --------------------

# Data Processing: Pivot Table for Average Price
pivot_table_avg_price = pasar_mini_df.pivot_table(
    values='price', 
    index='item_group', 
    columns='state', 
    aggfunc='mean'
)

# Section Objective Header
st.markdown("""
<div style="background: linear-gradient(90deg, #333333 0%, #555555 100%); 
            padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>Objective:</strong> To compare average item prices across item groups and states in order to identify regional and category-based price variations.
</div>
""", unsafe_allow_html=True)

# Expander (Closed by default)
with st.expander("Average Price by Item Group and State Heatmap", expanded=False):
    
    # Create the Interactive Heatmap
    fig_heat = px.imshow(
        pivot_table_avg_price, 
        aspect="auto",
        title='Heatmap: Price Variation by Region & Group',
        labels={'x':'State', 'y':'Item Group', 'color':'Avg Price (RM)'},
        color_continuous_scale='RdBu_r', # Red for High, Blue for Low
        color_continuous_midpoint=pasar_mini_df['price'].mean()
    )

    fig_heat.update_xaxes(side="top")
    fig_heat.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        margin=dict(t=100) # Space for the top x-axis labels
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # --- Insight Summary ---
    st.markdown("### Regional Price Analysis Insights")
    st.info("""
    * **High Expenditure Baseline:** 'Susu dan Barangan Bayi' represents the highest expenditure nationwide, peaking at **RM 26.39 in Sarawak**.
    * **Volatility in Beverages:** The 'Minuman' category shows the most extreme spread, ranging from **RM 6.95 (Sarawak)** to a high of **RM 16.56 (Johor)**.
    * **East vs. West Logistics:** East Malaysia reports higher averages for packaged and dry goods, while Federal Territories like **Putrajaya and KL** show the highest costs for 'Barangan Segar' (Fresh Produce) at **RM 15.37**.
    * **Data Availability:** The presence of gaps (NaN values) for cleaning products in certain regions highlights varying levels of product availability across the Malaysian retail landscape.
    """)

    # --- Data Table (Pivot) ---
    st.markdown("#### Cross-tabulation: Average Price (RM)")
    st.dataframe(pivot_table_avg_price.style.format("{:.2f}").highlight_null(color='lightgrey'), use_container_width=True)

st.markdown("---")








st.markdown("---")
