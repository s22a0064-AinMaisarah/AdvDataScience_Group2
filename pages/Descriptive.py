import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Descriptive Across Price Among Pasar Mini",
    layout="wide"
)

# --------------------
# 2. LOAD DATA
# --------------------
@st.cache_data
def load_data():
    # Creating a dummy file check to prevent crash if file is missing
    file_path = "dataset/pasar_mini_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        return None

pasar_mini_df = load_data()

# ---------------------------------------------------------
# 3. CUSTOM STYLES (CSS)
# ---------------------------------------------------------
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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. HEADER SECTION
# ---------------------------------------------------------
st.markdown('<div class="center-title">Descriptive Across Price Among Pasar Mini</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Nurul Ain Maisarah Binti Hamidin | S22A0064</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. DATA VISUALIZATION SECTION
# ---------------------------------------------------------
if pasar_mini_df is not None:
    with st.expander("🔍 View Raw Dataset", expanded=False):
        st.dataframe(pasar_mini_df, use_container_width=True)

    # ---------------------------------------------------------
    # 6. KPI METRICS & INSIGHTS
    # ---------------------------------------------------------
    st.write("### Descriptive Summary")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(
            label="Lowest Price Item",
            value="RM 0.50",
            help="Serbuk Kari Ayam dan Daging Adabi."
        )

    with m_col2:
        st.metric(
            label="Highest Price Item",
            value="RM 498.00",
            help="Bawang Besar Import (India) | 2025-12-19."
        )

    with m_col3:
        st.metric(
            label="Most Sales Item",
            value="minyak masak tulen",
            help="Average Price RM 18.77."
        )

    with m_col4:
        st.metric(
            label="Most Sales Premise ID",
            value="1641",
            help="Pasar Raya Kifarah Fresh Mart."
        )

    st.markdown("---")
    
# ---------------------------------------------------------
# 7. TIME SERIES ANALYSIS (Average Price Trend)
# ---------------------------------------------------------

with st.expander("📈 Average Price Trends Over Time", expanded=False):
    
    # 1. Calculate the average price per date
    # Ensure 'date' is datetime and 'price' is numeric
    pasar_mini_df['date'] = pd.to_datetime(pasar_mini_df['date'])
    average_price_per_date = pasar_mini_df.groupby('date')['price'].mean().reset_index()

    st.subheader("Interactive Price Trend Analysis")

    # 2. Create the interactive line plot
    fig_line = px.line(
        average_price_per_date,
        x='date',
        y='price',
        markers=True,
        title='Average Price Fluctuations in Pasar Mini',
        labels={'date': 'Date', 'price': 'Average Price (RM)'},
        line_shape='linear', 
        color_discrete_sequence=["#FF4081"], # Pink color as per your original font color
        hover_data={
            'date': '|%Y-%m-%d', 
            'price': ':.2f'
        }
    )

    fig_line.update_layout(
        hovermode='x unified',
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig_line.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

    # Display chart in Streamlit
    st.plotly_chart(fig_line, use_container_width=True)

    # 3. Layout for Table and Insights
    col_table, col_insight = st.columns([1, 1.2])

    with col_table:
        st.write("**Top 10 Price Records**")
        st.dataframe(average_price_per_date.head(10), use_container_width=True, hide_index=True)

    with col_insight:
        st.write("**📊 Analysis & Insights**")
        
        # Calculate some dynamic insights
        max_price = average_price_per_date['price'].max()
        min_price = average_price_per_date['price'].min()
        latest_avg = average_price_per_date['price'].iloc[-1]
        
        st.info(f"""
        - **Price Volatility:** The average price across Pasar Mini premises shows a fluctuation between **RM {min_price:.2f}** and **RM {max_price:.2f}**.
        - **Current Trend:** The most recent recorded average price stands at **RM {latest_avg:.2f}**.
        - **Market Behavior:** Spikes in the line chart often correlate with the arrival of imported goods (e.g., Bawang Besar Import) or supply chain shifts.
        - **Stability:** Periods where the line is flatter indicate steady pricing for essential household items like *Minyak Masak Buruh*.
        """)

st.markdown("---")
