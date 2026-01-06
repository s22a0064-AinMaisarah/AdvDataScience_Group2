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
        <div class="metric-label">📦 Top Item Category</div>
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

st.markdown("---")









st.markdown("---")
