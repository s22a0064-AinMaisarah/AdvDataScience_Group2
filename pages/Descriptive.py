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
st.subheader("Key Dataset Metrics")
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
# 8. Visualisation: Central Tendency 
# --------------------
price_mean = pasar_mini_df['price'].mean()
price_median = pasar_mini_df['price'].median()
price_mode = pasar_mini_df['price'].mode()[0] if not pasar_mini_df['price'].mode().empty else 0
price_skew = pasar_mini_df['price'].skew()
price_kurt = pasar_mini_df['price'].kurt()
price_count = len(pasar_mini_df)
max_price = pasar_mini_df['price'].max()

st.markdown("""
<div style="background: linear-gradient(90deg, #764ba2 0%, #4facfe 100%); padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong> Objective:</strong> To identify typical price levels (Mean, Median, Mode) within the dataset.
</div>
""", unsafe_allow_html=True)

with st.expander("MEASURES OF CENTRAL TENDENCY FOR PRICE", expanded=False):
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
        font=dict(color="white")
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- Summary Section ---
    st.markdown("### Statistical Summary")
    
    # Using a container for the point summary
    st.info(f"""
    * **Data Volume:** A total of **{price_count:,}** price points were analyzed.
    * **Typical Values:** The **Mean** price is **RM {price_mean:.2f}**, which is higher than the **Median (RM {price_median:.2f})** and **Mode (RM {price_mode:.2f})**.
    * **Distribution Shape:** A **Skewness of {price_skew:.2f}** indicates a **Right-Skewed** distribution. This means a few expensive items pull the average up, while most items are priced below RM 10.00.
    * **Price Extremes:** A high **Kurtosis of {price_kurt:.2f}** confirms a "heavy-tailed" distribution with significant outliers, reaching a maximum value of **RM {max_price:.2f}**.
    """)

    # --- Data Table ---
    st.markdown("#### Detailed Metrics")
    ct_df = pd.DataFrame({
        'Measure': ['Total Count', 'Mean', 'Median', 'Mode', 'Skewness', 'Kurtosis'],
        'Value': [f"{price_count:,}", f"RM {price_mean:.2f}", f"RM {price_median:.2f}", f"RM {price_mode:.2f}", f"{price_skew:.2f}", f"{price_kurt:.2f}"]
    })
    st.table(ct_df)

st.markdown("---")
