import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------
# 1. Load Data
# --------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/refs/heads/main/dataset/pasar_mini_data_updated.csv"
    df = pd.read_csv(url)
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    return df

# Handle data loading gracefully
try:
    pasar_mini_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --------------------
# 2. Header Styling & Section
# --------------------
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

st.markdown('<div class="center-title">Descriptive Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Nurul Ain Maisarah Binti Hamidin | S22A0064</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# --------------------
# 3. Dataset Preview (Glow Expander)
# --------------------
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

with st.expander("🔍 CLICK TO REVEAL DATASET PREVIEW", expanded=False):
    st.write("### Previewing First 5 Rows")
    st.dataframe(pasar_mini_df.head(), use_container_width=True)

# --------------------
# 4. KPI Metrics
# --------------------
st.markdown("""
<style>
.metric-card {
    background: #ffffff; border-radius: 12px; padding: 15px;
    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    border-bottom: 4px solid #ddd; transition: 0.2s;
}
.metric-card:hover { transform: translateY(-3px); }
.m-max { border-color: #FF4B4B; }
.m-min { border-color: #00CC96; }
.m-top { border-color: #636EFA; }
.m-cat { border-color: #AB63FA; }
.metric-label { font-size: 0.8rem; color: #666; font-weight: 700; }
.metric-value { font-size: 1.2rem; font-weight: 800; color: #31333F; }
.metric-help { font-size: 0.7rem; color: #999; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.subheader("Key Dataset Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card m-max"><div class="metric-label">Max Price</div><div class="metric-value">RM 498.00</div><div class="metric-help">Bawang Besar Import<br>2025-12-19</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card m-min"><div class="metric-label">Min Price</div><div class="metric-value">RM 0.50</div><div class="metric-help">Serbuk Kari Adabi<br>2025-12-08</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card m-top"><div class="metric-label">Top Premise</div><div class="metric-value">1,641</div><div class="metric-help">Kifarah Fresh Mart</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card m-cat"><div class="metric-label">Top Category</div><div class="metric-value">67,098</div><div class="metric-help">Barangan Berbungkus</div></div>', unsafe_allow_html=True)

# --------------------
# 5. Objectives
# --------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Descriptive Objectives")
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1.2rem; border-radius: 12px; color: white;">
    To descriptively analyze price patterns, distribution characteristics across time, location, and item classifications among Pasar Mini.
</div>
""", unsafe_allow_html=True)

# --------------------
# 6. Visualisation: Average Price Over Time
# --------------------
st.markdown("---")
st.markdown("### Visual Analysis")

# Data Processing
avg_price = pasar_mini_df.groupby('date')['price'].mean().reset_index()

# Objective Card for Chart
st.markdown("""
<div style="background: linear-gradient(90deg, #FF4081 0%, #764ba2 100%); padding: 10px 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
    <strong>🎯 Chart Objective:</strong> To examine trends and changes in average item prices over time.
</div>
""", unsafe_allow_html=True)

with st.expander("AVERAGE PRICE OVER TIME ANALYSIS", expanded=True):
    fig = px.line(avg_price, x='date', y='price', markers=True,
                 labels={'date': 'Date', 'price': 'Average Price (RM)'},
                 line_shape='spline', color_discrete_sequence=['#FF4081'])
    
    fig.update_layout(hovermode='x unified', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                      font=dict(color="#FF4081"), margin=dict(t=30))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📋 Top 10 Date Summaries")
    st.dataframe(avg_price.head(10), use_container_width=True)
    import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# --------------------
# 1. Calculation Logic
# --------------------
price_mean = pasar_mini_df['price'].mean()
price_median = pasar_mini_df['price'].median()
price_mode = pasar_mini_df['price'].mode()[0] if not pasar_mini_df['price'].mode().empty else 0

# Prepare data for plotting
measures = ['Mean', 'Median', 'Mode']
values = [price_mean, price_median, price_mode]
colors = ['#4facfe', '#764ba2', '#00f2fe'] # Matching your dashboard palette

# --------------------
# 2. Visual Style for this Section
# --------------------
st.markdown("""
<style>
/* Section-specific glowing expander */
.ct-expander {
    border: 2px solid #764ba2 !important; 
    border-radius: 15px !important;
}

@keyframes pulse-purple {
    0% { box-shadow: 0 0 5px #764ba2; }
    50% { box-shadow: 0 0 15px #764ba2; }
    100% { box-shadow: 0 0 5px #764ba2; }
}

.ct-glow {
    animation: pulse-purple 4s infinite;
    border-radius: 15px !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# 3. Objective Header
# --------------------
st.markdown("""<div style="background: linear-gradient(90deg, #764ba2 0%, #4facfe 100%); padding: 10px; border-radius: 10px; color: white; margin-bottom: 10px;">
    <strong>🎯 Objective:</strong> Identify typical price levels (Mean, Median, Mode).</div>""", unsafe_allow_html=True)

with st.expander("📊 CLICK TO VIEW: MEASURES OF CENTRAL TENDENCY", expanded=False):
    p_mean, p_median = pasar_mini_df['price'].mean(), pasar_mini_df['price'].median()
    p_mode = pasar_mini_df['price'].mode()[0]
    
    fig2 = go.Figure(data=[go.Bar(x=['Mean', 'Median', 'Mode'], y=[p_mean, p_median, p_mode], marker_color=['#4facfe', '#764ba2', '#00f2fe'])])
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
    st.plotly_chart(fig2, use_container_width=True)
        
        # Create the interactive bar chart
        fig = go.Figure(data=[go.Bar(
            x=measures,
            y=values,
            marker_color=colors,
            text=[f'RM {val:.2f}' for val in values],
            textposition='auto',
            hoverinfo='text',
            hovertext=[f'<b>{m}</b><br>Value: RM {v:.2f}' for m, v in zip(measures, values)]
        )])

        fig.update_layout(
            title_text="Central Tendency Analysis",
            xaxis_title="Statistical Measure",
            yaxis_title="Price (RM)",
            title_x=0.5,
            font=dict(family="Inter, sans-serif", size=13, color="#764ba2"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20),
            height=350
        )

        fig.update_xaxes(showline=True, linewidth=1, linecolor='rgba(150,150,150,0.3)')
        fig.update_yaxes(showline=True, linewidth=1, linecolor='rgba(150,150,150,0.3)')

        # Display chart
        st.plotly_chart(fig, use_container_width=True)

        # Display the Summary Table
        st.markdown("#### Summary Statistics Table")
        central_tendency_df = pd.DataFrame({
            'Measure': ['Mean (Average)', 'Median (Middle Value)', 'Mode (Most Frequent)'],
            'Value (RM)': [f"RM {price_mean:.2f}", f"RM {price_median:.2f}", f"RM {price_mode:.2f}"]
        })
        st.table(central_tendency_df)

st.markdown("---")
