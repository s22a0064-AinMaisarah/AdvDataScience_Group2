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

# ---------------------------------------------------------
# 11. DISTRIBUTION SHAPE ANALYSIS
# ---------------------------------------------------------

with st.expander("📐 Distribution Shape (Skewness & Kurtosis)", expanded=False):
    
    # 1. Calculate statistical measures
    price_skewness = pasar_mini_df['price'].skew()
    price_kurtosis = pasar_mini_df['price'].kurtosis()

    distribution_shape_df = pd.DataFrame({
        'Measure': ['Skewness', 'Kurtosis'],
        'Value': [price_skewness, price_kurtosis]
    })

    st.subheader("Interactive Distribution Shape Analysis")

    # 2. Create the Bar Chart
    fig_shape = px.bar(
        distribution_shape_df,
        x='Measure',
        y='Value',
        color='Measure',
        color_discrete_sequence=px.colors.qualitative.Dark24,
        title="Distribution Shape: Price Asymmetry & Peakedness",
        labels={'Measure': 'Measure', 'Value': 'Value'},
        text='Value'
    )

    fig_shape.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_shape.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    fig_shape.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig_shape.update_yaxes(showline=True, linewidth=1, linecolor='black')

    # Display chart in Streamlit
    st.plotly_chart(fig_shape, use_container_width=True)

    # 3. Layout for Table and Insights
    col_tbl, col_ins = st.columns([1, 1.2])

    with col_tbl:
        st.write("**Shape Statistics Table**")
        st.dataframe(distribution_shape_df, use_container_width=True, hide_index=True)

    with col_ins:
        st.write("**💡 Interpreting the Shape**")
        
        # Determine Skewness Note
        skew_desc = "Highly Positive" if price_skewness > 1 else "Moderate"
        
        # Determine Kurtosis Note
        kurt_desc = "Leptokurtic (Heavy tails/Outliers)" if price_kurtosis > 3 else "Platykurtic (Flat)"

        st.info(f"""
        - **Skewness ({price_skewness:.3f}):** A value > 1 indicates a **{skew_desc}** right-skew. This confirms that the majority of items are cheap, but the "tail" of the graph is stretched toward the expensive items.
        - **Kurtosis ({price_kurtosis:.3f}):** Since your kurtosis is likely high, it is **{kurt_desc}**. This indicates that the dataset has frequent outliers (extreme price differences) rather than a smooth bell curve.
        """)

# --- Add the divider as requested ---
st.markdown("---")

# ---------------------------------------------------------
# 12. PRICE BINNED DISTRIBUTION ANALYSIS
# ---------------------------------------------------------

with st.expander("📊 Price Range Distribution (Binned)", expanded=False):
    
    # 1. Manual Histogram Calculation
    counts, bin_edges = np.histogram(pasar_mini_df['price'], bins=50)

    bins_df = pd.DataFrame({
        'price_bin_start': bin_edges[:-1],
        'price_bin_end': bin_edges[1:],
        'count': counts
    })

    # 2. Calculate percentages and cumulative totals
    total_entries = bins_df['count'].sum()
    bins_df['percentage'] = (bins_df['count'] / total_entries) * 100
    bins_df['cumulative_count'] = bins_df['count'].cumsum()
    bins_df['cumulative_percentage'] = bins_df['percentage'].cumsum()

    # Format bin labels for the X-axis
    bins_df['price_range'] = bins_df.apply(
        lambda row: f"RM{row['price_bin_start']:.2f}-{row['price_bin_end']:.2f}",
        axis=1
    )

    st.subheader("Frequency Distribution of Prices")

    # 3. Create the Interactive Bar Chart
    fig_bins = px.bar(
        bins_df,
        x='price_range',
        y='count',
        color='percentage',
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Price Density: Item Counts per Price Bracket',
        labels={
            'price_range': 'Price Range (RM)',
            'count': 'Number of Items',
            'percentage': 'Market Share (%)'
        },
        hover_data={
            'count': True,
            'percentage': ':.2f',
            'cumulative_percentage': ':.2f'
        }
    )

    fig_bins.update_layout(
        title_x=0.5,
        font=dict(family="Arial, sans-serif", size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45,
        height=600
    )

    st.plotly_chart(fig_bins, use_container_width=True)

    st.divider()

    # 4. Data Table and Insights
    col_tbl, col_ins = st.columns([1.2, 1])

    with col_tbl:
        st.write("**Price Binning Table (Top 10 Ranges)**")
        # Showing the top ranges where most items fall
        display_bins = bins_df.sort_values('count', ascending=False).head(10)
        st.dataframe(display_bins[['price_range', 'count', 'percentage', 'cumulative_percentage']], 
                     use_container_width=True, hide_index=True)

    with col_ins:
        st.write("**🔍 Distribution Insights**")
        
        # Identify the peak bin
        peak_bin = bins_df.loc[bins_df['count'].idxmax()]
        
        st.success(f"""
        - **Market Concentration:** The most common price range is **{peak_bin['price_range']}**, containing **{peak_bin['count']}** different items.
        - **Density Analysis:** A tall spike at the beginning of the chart confirms that your Pasar Mini is heavily focused on **low-cost daily essentials**.
        - **The Long Tail:** The very short bars on the far right represent "Luxury" or "Bulk" items that are infrequent but high in value.
        - **Inventory Strategy:** This distribution suggests a high-volume, low-margin business model common in local mini markets.
        """)

# Final Footer Divider
st.markdown("---")

# ---------------------------------------------------------
# 13. CATEGORICAL MODE ANALYSIS
# ---------------------------------------------------------

with st.expander("🏷️ Most Frequent Categorical Values (Modes)", expanded=False):
    
    # 1. Data Preparation - Calculate modes for specific categorical columns
    # Adjust these column names to match your actual dataset columns
    cat_columns = ['item_name', 'premise_name', 'item_category'] 
    # Check which columns actually exist in the dataframe to avoid errors
    existing_cats = [c for c in cat_columns if c in pasar_mini_df.columns]
    
    modes_list = []
    for col in existing_cats:
        mode_val = pasar_mini_df[col].mode()
        if not mode_val.empty:
            modes_list.append({'Column': col, 'Mode': mode_val[0]})
            
    selected_modes_df = pd.DataFrame(modes_list)

    if not selected_modes_df.empty:
        st.subheader("Top Frequent Categories")

        # 2. Create the interactive bar chart
        fig_mode = px.bar(
            selected_modes_df,
            x='Column',
            y='Mode',
            color='Column',
            title='Most Frequent Values per Category',
            labels={'Column': 'Categorical Column', 'Mode': 'Most Frequent Value'},
            text='Mode'
        )

        fig_mode.update_layout(
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )

        fig_mode.update_xaxes(showline=True, linewidth=1, linecolor='black')
        # Hide Y-axis labels if they are names/text, otherwise it looks cluttered
        fig_mode.update_yaxes(showticklabels=False, showline=True, linewidth=1, linecolor='black')

        st.plotly_chart(fig_mode, use_container_width=True)

        # 3. Layout for Table and Insights
        st.divider()
        col_tbl, col_ins = st.columns([1, 1.2])

        with col_tbl:
            st.write("**Mode Summary Table**")
            st.dataframe(selected_modes_df, use_container_width=True, hide_index=True)

        with col_ins:
            st.write("**📝 Categorical Insights**")
            
            # Extract specific modes for the text explanation
            top_item = selected_modes_df.loc[selected_modes_df['Column'] == 'item_name', 'Mode'].values[0] if 'item_name' in existing_cats else "N/A"
            top_cat = selected_modes_df.loc[selected_modes_df['Column'] == 'item_category', 'Mode'].values[0] if 'item_category' in existing_cats else "N/A"

            st.success(f"""
            - **Dominant Item:** The most frequently appearing product in this record set is **{top_item}**. 
            - **Primary Category:** **{top_cat}** is the most well-represented category, indicating where the bulk of inventory data is focused.
            - **Business Interpretation:** Identifying the mode helps stakeholders understand which items or premises are over-represented in the survey, ensuring that price averages aren't biased toward a single dominant product.
            """)
    else:
        st.warning("No categorical columns found to calculate modes.")

# --- Final Page Divider ---
st.markdown("---")
st.caption("Dashboard Analysis Complete | Data Source: pasar_mini_data.csv")

# ---------------------------------------------------------
# 14. DECEMBER CUMULATIVE VOLUME ANALYSIS
# ---------------------------------------------------------

with st.expander("📅 December 2025 Reporting Volume", expanded=False):
    
    # 1. Filter and Prepare Data
    pasar_mini_df['date'] = pd.to_datetime(pasar_mini_df['date'])
    
    filtered_df_date = pasar_mini_df[
        (pasar_mini_df['date'].dt.year == 2025) & 
        (pasar_mini_df['date'].dt.month == 12) & 
        (pasar_mini_df['date'].dt.day <= 22)
    ].copy()

    if not filtered_df_date.empty:
        # Calculate frequency and average price
        date_counts = filtered_df_date['date'].value_counts().sort_index().reset_index()
        date_counts.columns = ['date', 'count']
        
        avg_price_date = filtered_df_date.groupby('date')['price'].mean().reset_index()
        avg_price_date.rename(columns={'price': 'average_price'}, inplace=True)
        
        # Merge and calculate cumulative stats
        date_counts = date_counts.merge(avg_price_date, on='date', how='left')
        date_counts['cumulative_count'] = date_counts['count'].cumsum()
        date_counts['cumulative_percentage'] = (date_counts['cumulative_count'] / len(filtered_df_date)) * 100

        st.subheader("Cumulative Entry Tracking (Dec 1st - 22nd)")

        # 2. Create Interactive Line Chart
        fig_volume = px.line(
            date_counts, x='date', y='cumulative_count', markers=True,
            title='Growth of Entries Over Time (Pasar Mini)',
            labels={'date': 'Date', 'cumulative_count': 'Total Entries to Date'},
            hover_data={'date': '|%Y-%m-%d', 'count': True, 'average_price': ':.2f'}
        )

        # 3. Add Star Markers for Highlighted Weekly Dates
        specific_dates = ['2025-12-01', '2025-12-08', '2025-12-15', '2025-12-22']
        
        for d_str in specific_dates:
            s_date = pd.to_datetime(d_str)
            if s_date in date_counts['date'].values:
                row = date_counts[date_counts['date'] == s_date].iloc[0]
                fig_volume.add_scatter(
                    x=[s_date], y=[row['cumulative_count']],
                    mode='markers', marker=dict(size=12, color='red', symbol='star'),
                    name=f"Milestone: {s_date.strftime('%b %d')}",
                    hovertext=(f"<b>Date:</b> {d_str}<br><b>Daily Count:</b> {row['count']}<br>"
                               f"<b>Cumulative:</b> {row['cumulative_count']}<br>"
                               f"<b>Avg Price:</b> RM{row['average_price']:.2f}")
                )

        fig_volume.update_layout(
            hovermode='x unified', title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12, color="#2ECC71"),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=80)
        )
        
        fig_volume.update_xaxes(tickformat="%d %b", gridcolor='LightGrey')
        fig_volume.update_yaxes(gridcolor='LightGrey')

        st.plotly_chart(fig_volume, use_container_width=True)

        # 4. Insights and Table
        st.divider()
        col_t, col_i = st.columns([1, 1.2])
        
        with col_t:
            st.write("**Daily Reporting Summary**")
            st.dataframe(date_counts[['date', 'count', 'cumulative_count']].head(10), 
                         use_container_width=True, hide_index=True)
            
        with col_i:
            st.write("**📈 Volume Insights**")
            total_period_entries = date_counts['count'].sum()
            st.success(f"""
            - **Total Period Volume:** Between Dec 1st and Dec 22nd, a total of **{total_period_entries:,}** entries were recorded.
            - **Growth Pattern:** The cumulative line chart shows how quickly data is being aggregated. A steeper slope indicates a high-activity recording day.
            - **Milestone Highlight:** The red stars represent the start of each week, allowing you to compare weekly performance directly.
            - **Data Freshness:** The latest average price in this specific window provides a real-time snapshot of grocery costs during the holiday month.
            """)
    else:
        st.warning("No data found for the specified range in December 2025.")

# Final Section Divider
st.markdown("---")

# ---------------------------------------------------------
# 15. ITEM FREQUENCY & CUMULATIVE ANALYSIS
# ---------------------------------------------------------

with st.expander("🍎 Item Prevalence & Cumulative Analysis", expanded=False):
    
    # 1. Calculate Frequency and Proportions
    # Note: Using 'item' column as per your code
    if 'item' in pasar_mini_df.columns:
        item_counts = pasar_mini_df['item'].value_counts().reset_index()
        item_counts.columns = ['item', 'count']

        # Calculate average price per item
        avg_price_item = pasar_mini_df.groupby('item')['price'].mean().reset_index()
        avg_price_item.rename(columns={'price': 'average_price'}, inplace=True)

        # Merge and sort
        item_counts = item_counts.merge(avg_price_item, on='item', how='left')
        item_counts = item_counts.sort_values(by='count', ascending=False)

        # Cumulative Calculations
        total_rows = len(pasar_mini_df)
        item_counts['percentage'] = (item_counts['count'] / total_rows) * 100
        item_counts['cumulative_count'] = item_counts['count'].cumsum()
        item_counts['cumulative_percentage'] = item_counts['percentage'].cumsum()

        st.subheader("Top 15 Most Frequent Items")

        # 2. Create Interactive Bar Chart
        fig_item = px.bar(
            item_counts.head(15), 
            x='item',
            y='count',
            color='item',
            title='Frequency Distribution (Top 15 Items)',
            labels={'item': 'Item Name', 'count': 'Number of Entries'},
            hover_data={
                'percentage': ':.2f%',
                'cumulative_count': True,
                'cumulative_percentage': ':.2f%',
                'average_price': ':.2f'
            }
        )

        fig_item.update_layout(
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12, color="#4CAF50"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig_item, use_container_width=True)

        # 3. Table and Insights
        st.divider()
        col_t, col_i = st.columns([1.3, 1])

        with col_t:
            st.write("**Cumulative Frequency Table (Top 15)**")
            st.dataframe(item_counts.head(15), use_container_width=True, hide_index=True)

        with col_i:
            st.write("**📊 Pareto Insights**")
            
            # Find how many items make up 80% of data (approx)
            top_80_count = len(item_counts[item_counts['cumulative_percentage'] <= 85])
            
            st.info(f"""
            - **Data Density:** The top 15 items represent a significant portion of your total market records.
            - **Cumulative Reach:** You can see exactly when the **{item_counts['item'].iloc[0]}** is combined with other top items, how much of the total dataset they cover.
            - **Average Price Context:** Hovering over the bars reveals the average price, allowing you to see if the most frequent items are also the most affordable.
            - **Market Focus:** This list helps identify which products are the "staples" of Pasar Mini inventories.
            """)
    else:
        st.warning("Column 'item' not found in the dataset.")

# Final Section Divider
st.markdown("---")

# ---------------------------------------------------------
# 16. ITEM GROUP CUMULATIVE ANALYSIS
# ---------------------------------------------------------

with st.expander("📦 Item Group Frequency & Cumulative Analysis", expanded=False):
    
    # 1. Calculate Frequency and Proportions for Item Groups
    if 'item_group' in pasar_mini_df.columns:
        item_group_counts = pasar_mini_df['item_group'].value_counts().reset_index()
        item_group_counts.columns = ['item_group', 'count']

        # Calculate average price per group
        avg_price_group = pasar_mini_df.groupby('item_group')['price'].mean().reset_index()
        avg_price_group.rename(columns={'price': 'average_price'}, inplace=True)

        # Merge and sort
        item_group_counts = item_group_counts.merge(avg_price_group, on='item_group', how='left')
        item_group_counts = item_group_counts.sort_values(by='count', ascending=False)

        # Cumulative Calculations
        total_rows_group = len(pasar_mini_df)
        item_group_counts['percentage'] = (item_group_counts['count'] / total_rows_group) * 100
        item_group_counts['cumulative_count'] = item_group_counts['count'].cumsum()
        item_group_counts['cumulative_percentage'] = item_group_counts['percentage'].cumsum()

        st.subheader("Distribution by Product Group")

        # 2. Create Interactive Bar Chart
        fig_group = px.bar(
            item_group_counts, 
            x='item_group',
            y='count',
            color='item_group',
            title='Frequency and Cumulative Weight by Item Group',
            labels={'item_group': 'Item Group', 'count': 'Number of Entries'},
            hover_data={
                'percentage': ':.2f%',
                'cumulative_count': True,
                'cumulative_percentage': ':.2f%',
                'average_price': ':.2f'
            }
        )

        fig_group.update_layout(
            title_x=0.5,
            font=dict(family="Arial, sans-serif", size=12, color="#4CAF50"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45,
            showlegend=False
        )

        st.plotly_chart(fig_group, use_container_width=True)

        # 3. Table and Insights
        st.divider()
        col_t, col_i = st.columns([1.3, 1])

        with col_t:
            st.write("**Cumulative Frequency Table (Top 10 Groups)**")
            st.dataframe(item_group_counts.head(10), use_container_width=True, hide_index=True)

        with col_i:
            st.write("**📊 Group Insights**")
            
            top_group = item_group_counts['item_group'].iloc[0]
            top_group_pct = item_group_counts['percentage'].iloc[0]
            
            st.info(f"""
            - **Leading Category:** The **{top_group}** group is the most prevalent, accounting for **{top_group_pct:.2f}%** of all recorded entries.
            - **Inventory Breadth:** By looking at the cumulative percentage, you can see how many groups are required to cover 90% of your total data.
            - **Price Comparison:** Note how the `average_price` varies between groups—this helps identify which categories are "High Volume, Low Cost" vs "Low Volume, High Cost."
            - **Data Symmetry:** A steep decline in bar heights suggests that your data collection is focused heavily on a few specific departments.
            """)
    else:
        st.warning("Column 'item_group' not found in the dataset.")

# Final Section Divider
st.markdown("---")

