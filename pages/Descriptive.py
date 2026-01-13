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

# ---------------------------------------------------------
# 8. CENTRAL TENDENCY ANALYSIS (Mean, Median, Mode)
# ---------------------------------------------------------

with st.expander("📊 Measures of Central Tendency", expanded=False):
    
    # 1. Calculate measures of central tendency
    price_mean = pasar_mini_df['price'].mean()
    price_median = pasar_mini_df['price'].median()
    # Mode can return multiple values, so we take the first one
    price_mode = pasar_mini_df['price'].mode()[0] if not pasar_mini_df['price'].mode().empty else 0

    st.subheader("Statistical Price Distribution")

    # 2. Prepare data for plotting
    measures = ['Mean (Average)', 'Median (Middle)', 'Mode (Most Frequent)']
    values = [price_mean, price_median, price_mode]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

    # 3. Create the interactive bar chart
    fig_central = go.Figure(data=[go.Bar(
        x=measures,
        y=values,
        marker_color=colors,
        text=[f'RM {val:.2f}' for val in values],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f'<b>{m}:</b> RM {v:.2f}' for m, v in zip(measures, values)]
    )])

    fig_central.update_layout(
        title_text="Price Distribution: Central Tendency",
        xaxis_title="Statistical Measure",
        yaxis_title="Price (RM)",
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig_central.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig_central.update_yaxes(showline=True, linewidth=1, linecolor='black')

    # Display chart in Streamlit
    st.plotly_chart(fig_central, use_container_width=True)

    # 4. Layout for Table and Insights
    col_tbl, col_ins = st.columns([1, 1.2])

    with col_tbl:
        st.write("**Central Tendency Summary**")
        central_tendency_df = pd.DataFrame({
            'Measure': ['Mean', 'Median', 'Mode'],
            'Value (RM)': [f"{price_mean:.2f}", f"{price_median:.2f}", f"{price_mode:.2f}"]
        })
        st.table(central_tendency_df)

    with col_ins:
        st.write("**📝 Statistical Insights**")
        
        # Determine skewness based on mean vs median
        skew_type = "Right-Skewed (Positive)" if price_mean > price_median else "Left-Skewed (Negative)"
        skew_note = "expensive outliers (like the RM 498.00 item) are pulling the average up." if price_mean > price_median else "cheaper items are pulling the average down."

        st.success(f"""
        - **Data Distribution:** The distribution is **{skew_type}**. This means {skew_note}
        - **Mean (RM {price_mean:.2f}):** This is the mathematical average. It is sensitive to extreme prices.
        - **Median (RM {price_median:.2f}):** The 50th percentile. Half of your grocery items are cheaper than this value, and half are more expensive.
        - **Mode (RM {price_mode:.2f}):** This is the most common price found in the dataset.
        """)

st.markdown("---")

# ---------------------------------------------------------
# 9. DISPERSION ANALYSIS (Variance & Standard Deviation)
# ---------------------------------------------------------

with st.expander("📉 Measures of Price Dispersion", expanded=False):
    
    # 1. Calculate dispersion measures
    # Standard Deviation: average distance from the mean
    price_std = pasar_mini_df['price'].std()
    # Variance: the squared standard deviation
    price_var = pasar_mini_df['price'].var()
    # Range: difference between max and min
    price_range = pasar_mini_df['price'].max() - pasar_mini_df['price'].min()

    st.subheader("Interactive Dispersion Analysis")

    # 2. Create the DataFrame for plotting
    price_stats_df = pd.DataFrame({
        'Measure': ['Std Deviation', 'Variance', 'Price Range'],
        'Value': [price_std, price_var, price_range]
    })

    # 3. Create the bar chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig_disp = px.bar(
        price_stats_df,
        x='Measure',
        y='Value',
        color='Measure',
        color_discrete_sequence=colors,
        title="Price Variance & Spread in Pasar Mini",
        labels={'Measure': 'Statistical Measure', 'Value': 'Value'},
        text='Value'
    )

    fig_disp.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_disp.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    fig_disp.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig_disp.update_yaxes(showline=True, linewidth=1, linecolor='black')

    # Display chart in Streamlit
    st.plotly_chart(fig_disp, use_container_width=True)

    # 4. Layout for Table and Insights
    col_tbl, col_ins = st.columns([1, 1.2])

    with col_tbl:
        st.write("**Dispersion Summary Table**")
        st.dataframe(price_stats_df, use_container_width=True, hide_index=True)

    with col_ins:
        st.write("**💡 Understanding the Spread**")
        
        st.warning(f"""
        - **Standard Deviation (RM {price_std:.2f}):** On average, prices deviate from the mean by this amount. A high value suggests that prices are very inconsistent across items.
        - **Variance ({price_var:.2f}):** This represents how far the price set is spread out from their average value. 
        - **Price Range (RM {price_range:.2f}):** This is the gap between the cheapest item (RM 0.50) and the most expensive (RM 498.00).
        - **Analysis:** Because the Range and Variance are quite high, it indicates a **highly diverse inventory** ranging from basic spices to high-value bulk imports.
        """)
        st.markdown("---")
# ---------------------------------------------------------
# 10. CUMULATIVE FREQUENCY & PERCENTILES
# ---------------------------------------------------------

with st.expander("📈 Cumulative Price Distribution & Percentiles", expanded=False):
    
    # 1. Data Preparation
    price_data = pasar_mini_df['price'].sort_values().reset_index(drop=True)
    total_count = len(price_data)
    cumulative_counts = price_data.value_counts(sort=False).sort_index().cumsum()
    cumulative_percentages = (cumulative_counts / total_count) * 100

    cumulative_df = pd.DataFrame({
        'price': cumulative_percentages.index,
        'cumulative_percentage': cumulative_percentages.values
    })

    # 2. Re-calculate stats for annotations
    p_min = price_data.min()
    p_max = price_data.max()
    p_median = price_data.median()
    q1 = price_data.quantile(0.25)
    q3 = price_data.quantile(0.75)

    st.subheader("Ogive Chart: Cumulative Percentage Analysis")

    # 3. Create the Figure
    fig_ogive = go.Figure()

    # Cumulative line
    fig_ogive.add_trace(go.Scatter(
        x=cumulative_df['price'], y=cumulative_df['cumulative_percentage'],
        mode='lines', name='Cumulative %', line=dict(color='#1f77b4', width=3)
    ))

    # Helper function for annotations
    def add_stat_line(fig, x_val, y_val, label, color, dash="dash"):
        fig.add_shape(type="line", x0=p_min, y0=y_val, x1=x_val, y1=y_val, line=dict(color=color, width=1, dash=dash))
        fig.add_shape(type="line", x0=x_val, y0=0, x1=x_val, y1=y_val, line=dict(color=color, width=1, dash=dash))
        fig.add_annotation(x=x_val, y=y_val, text=f"{label}: {x_val:.2f}", showarrow=True, 
                           arrowhead=2, bgcolor="white", bordercolor=color)

    # Add Median, Q1, Q3
    add_stat_line(fig_ogive, p_median, 50, "Median", "red")
    add_stat_line(fig_ogive, q1, 25, "Q1", "green", "dot")
    add_stat_line(fig_ogive, q3, 75, "Q3", "purple", "dot")

    fig_ogive.update_layout(
        xaxis_title='Price (RM)', yaxis_title='Cumulative Percentage (%)',
        hovermode='x unified', title_x=0.5, height=600,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_ogive, use_container_width=True)

    # 4. Detailed Summary & Insights
    st.divider()
    col_table, col_text = st.columns([1, 1.5])

    with col_table:
        st.write("**Descriptive Statistics Table**")
        st.dataframe(pasar_mini_df['price'].describe().to_frame(), use_container_width=True)

    with col_text:
        st.write("**🔍 Market Insights**")
        st.info(f"""
        - **The 75% Rule (Q3):** 75% of all items in your Pasar Mini dataset are priced below **RM {q3:.2f}**.
        - **The 25% Rule (Q1):** The bottom quarter of your inventory consists of very affordable items priced under **RM {q1:.2f}**.
        - **Interquartile Range (IQR):** The "middle 50%" of your prices fall between **RM {q1:.2f} and RM {q3:.2f}**. This is the core price range for most groceries.
        - **Max Outlier:** Notice how the curve flattens significantly after RM 100. This confirms that items like the RM 498.00 Bawang are rare outliers compared to the rest of the stock.
        """)
         st.markdown("---")

