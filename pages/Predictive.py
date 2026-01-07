import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="Predictive Analytics – Pasar Mini",
    layout="wide"
)

# --------------------
# LOAD DATA
# --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/pasar_mini_data_updated.csv")  # local path
    df['date'] = pd.to_datetime(df['date'])
    return df

pasar_mini_df = load_data()

# --------------------
# HEADER
# --------------------
st.markdown("<h1 style='text-align:center;'>📈 Predictive Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray;'>Forecasting Food Prices in Pasar Mini Markets</h4>", unsafe_allow_html=True)
st.markdown("---")

# --------------------
# OBJECTIVES
# --------------------
st.subheader("🎯 Objectives")
st.markdown("""
- To develop machine learning models for forecasting monthly food prices in Pasar Mini markets  
- To compare multiple predictive techniques based on performance metrics  
- To identify key factors influencing food price movements  
- To generate reliable forecasts for proactive planning in 2025  
""")

# --------------------
# PROBLEM STATEMENT
# --------------------
st.subheader("❗ Problem Statement")
st.markdown("""
Food prices in Pasar Mini markets fluctuate due to seasonal trends, geographic variation, and product characteristics.
Without predictive insights, retailers and policymakers may struggle to anticipate price changes and plan effectively.
This study applies predictive analytics to historical PriceCatcher data to forecast future food prices and support
data-driven decision-making.
""")

# --------------------
# DATASET PREVIEW
# --------------------
with st.expander("🔍 Dataset Preview"):
    st.dataframe(pasar_mini_df.head(), use_container_width=True)

# --------------------
# FEATURE ENGINEERING
# --------------------
predict_df = pasar_mini_df.copy()
predict_df["year"] = predict_df["date"].dt.year
predict_df["month"] = predict_df["date"].dt.month

monthly_price = (
    predict_df
    .groupby(["year","month","state_enc","district_enc",
              "item_group_enc","item_category_enc"])["price"]
    .mean()
    .reset_index()
)

# Sort BEFORE split
monthly_price = monthly_price.sort_values(["year","month"]).reset_index(drop=True)

features = ["year","month","state_enc","district_enc",
            "item_group_enc","item_category_enc"]
target = "price"

X = monthly_price[features]
y = monthly_price[target]

# --------------------
# TIME-BASED SPLIT
# --------------------
split_index = int(len(monthly_price) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------
# MODEL TRAINING
# --------------------
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_model = SVR(C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)

models = {
    "Random Forest": y_pred_rf,
    "Linear Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "SVR": y_pred_svr
}

# --------------------
# MODEL PERFORMANCE SUMMARY
# --------------------
with st.expander("📊 Model Performance Summary"):
    metrics = []
    for name, preds in models.items():
        metrics.append([
            name,
            mean_absolute_error(y_test, preds),
            np.sqrt(mean_squared_error(y_test, preds)),
            r2_score(y_test, preds)
        ])

    metrics_df = pd.DataFrame(metrics, columns=["Model","MAE","RMSE","R²"])
    st.dataframe(metrics_df, use_container_width=True)

    # Identify best model (lowest RMSE)
    best_model = metrics_df.sort_values("RMSE").iloc[0]

    st.markdown("### ✅ Best Model Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", best_model["Model"])
    col2.metric("Lowest RMSE", f"{best_model['RMSE']:.3f}")
    col3.metric("Highest R²", f"{best_model['R²']:.3f}")

# --------------------
# PREDICTIVE VISUALIZATIONS
# --------------------
st.subheader("🔮 Predictive Analytics & Forecast Visualizations")
st.markdown("""
This section presents the predictive analysis results, including monthly forecasted prices 
for Pasar Mini markets in 2025. Visualizations provide both trend analysis and comparative insights.
""")
    
# --------------------
# FORECAST 2025 (RF)
# --------------------
st.subheader("1. 📈 Forecasted Monthly Food Prices in Pasar Mini for 2025")

# Create DataFrame from your provided results
forecast_results = pd.DataFrame({
    "Year": [2025]*12,
    "Month": list(range(1,13)),
    "State": [0]*12,
    "District": [10]*12,
    "Item Group": [0]*12,
    "Item Category": [5]*12,
    "Predicted Price (RM)": [
        3.80, 3.85, 3.90, 3.95, 4.00, 4.05,
        4.10, 4.15, 4.20, 4.25, 4.30, 4.35
    ]
})

# Display table in Streamlit
st.dataframe(forecast_results, use_container_width=True)

# --------------------
# Forecasted Line Chart (Trend Over Time)
# --------------------

with st.expander("📉 Forecast Trend: Monthly Food Prices (2025)", expanded=True):
    st.subheader("📉 Forecasted Price Trend (Line Chart)")
    
    fig_line = px.line(
        forecast_results,
        x="Month",
        y="Predicted Price (RM)",
        markers=True,
        title="Forecasted Monthly Food Prices in Pasar Mini for 2025",
        labels={
            "Month": "Month",
            "Predicted Price (RM)": "Predicted Price (RM)"
        }
    )
    
    st.plotly_chart(fig_line, use_container_width=True)
    st.caption(
        "📌 This line chart shows the overall trend of forecasted food prices in Pasar Mini markets for 2025."
    )

# --------------------
# Monthly Forecast Bar Chart (Month to Month Comparison)
# --------------------
with st.expander("📊 Monthly Price Comparison (Bar Chart)", expanded=False):
    st.subheader("📊 Predicted Price Comparison (Bar Chart)")
    
    fig_bar = px.bar(
        forecast_results,
        x="Month",
        y="Predicted Price (RM)",
        text="Predicted Price (RM)",
        labels={"Month": "Month", "Predicted Price (RM)": "Predicted Price (RM)"},
        title="Predicted Monthly Food Prices in Pasar Mini for 2025"
    )
    fig_bar.update_traces(texttemplate="RM %{text:.2f}", textposition="outside")
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "📌 This bar chart provides a clear month-to-month comparison of predicted food prices."
    )

# --------------------
# ACTUAL VS PREDICTED
# --------------------
st.subheader("2. 🔍 Actual vs Predicted Comparison")

# Create scatter plots for each model
for name, preds in models.items():
    with st.expander(f"View Actual vs Predicted - {name}"):
        fig = px.scatter(
            x=y_test,
            y=preds,
            labels={"x": "Actual Price", "y": "Predicted Price"},
            title=f"{name}: Actual vs Predicted"
        )
        # Add perfect prediction line
        fig.add_shape(
            type="line",
            x0=y_test.min(),
            y0=y_test.min(),
            x1=y_test.max(),
            y1=y_test.max(),
            line=dict(color="red", width=2)
        )
        st.plotly_chart(fig, use_container_width=True)

# Add a **single interpretation expander** at the end
with st.expander("🔹 Overall Interpretation of Actual vs Predicted"):
    st.markdown("""
- **Random Forest:** Points closely follow the diagonal line, showing strong predictive accuracy for low and mid-range prices, with minor deviations at higher prices.  
- **Decision Tree:** Predictions align fairly well but form horizontal clusters due to grouping of similar prices.  
- **SVR & Linear Regression:** SVR captures the trend but struggles with high prices; Linear Regression shows a linear trend but wider errors at higher prices, indicating limited ability to capture non-linear patterns.  
- **Overall:** Random Forest handles complexity best, while simpler models have more difficulty with extreme or non-linear price patterns.
""")


# --------------------
# RESIDUAL PLOTS (ALL MODELS)
# --------------------
st.subheader("3. 📉 Residual Distribution")

for name, y_pred in models.items():
    residuals = y_test.values - y_pred

    # Create a DataFrame for residuals
    res_df = pd.DataFrame({"Residuals": residuals})

    # Create an expander for each model
    with st.expander(f"Residual Plot – {name}", expanded=False):
        fig = px.histogram(
            res_df,
            x="Residuals",
            nbins=30,
            opacity=0.75,
            title=f"Residual Distribution – {name}",
            labels={"Residuals": "Prediction Error (RM)"}
        )

        fig.update_layout(
            xaxis_title="Residuals",
            yaxis_title="Frequency",
            bargap=0.1
        )

        st.plotly_chart(fig, use_container_width=True)

# Add a **single interpretation expander** at the end
with st.expander("🔹 Overall Interpretation of Residual Analysis"):
        st.markdown("""
### 🔹 Residual Analysis
- **Random Forest:** Errors are small and centered around zero, showing accurate and balanced predictions for complex patterns.  
- **SVR:** Residuals slightly skewed, indicating occasional underprediction.  
- **Decision Tree & Linear Regression:** Decision Tree precise for most cases but shows some extreme errors; Linear Regression has wider, skewed residuals, reflecting consistent inaccuracies.  
""")

# --------------------
# FEATURE IMPORTANCE (RF & DT)
# --------------------
st.subheader("4. 🌟 Feature Importance")

with st.expander("Show Random Forest Feature Importance"):
    rf_imp = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    st.plotly_chart(px.bar(rf_imp, x="Feature", y="Importance",
                           title="Random Forest Feature Importance"), use_container_width=True)

with st.expander("Show Decision Tree Feature Importance"):
    dt_imp = pd.DataFrame({
        "Feature": features,
        "Importance": dt_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    st.plotly_chart(px.bar(dt_imp, x="Feature", y="Importance",
                           title="Decision Tree Feature Importance"), use_container_width=True)

    st.markdown("""
### 🔹 Feature Importance – Tree-Based Models
- **Random Forest & Decision Tree:** Product-related features are most important; item category is the strongest, followed by item group.  
- Geographic factors (state, district) play a minor role, while time (month, year) has minimal influence.  
- Random Forest balances influence across trees for stability, while Decision Tree relies on a single structure but both indicate that **what the product is matters more than when or where it is sold**.  
""")

# --------------------
# LINEAR REGRESSION COEFFICIENTS
# --------------------
st.subheader("5. 📐 Linear Regression Coefficients")

with st.expander("View Linear Regression Coefficients Chart"):
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr_model.coef_
    }).sort_values("Coefficient", ascending=False)

    fig = px.bar(
        coef_df,
        x="Feature",
        y="Coefficient",
        title="Linear Regression Feature Coefficients",
        labels={"Coefficient":"Coefficient Value", "Feature":"Features"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: display raw dataframe
    st.write("Data used for chart:")
    st.dataframe(coef_df)
    
    st.markdown("""
### 🔹 Feature Importance – Linear Regression
- Item-related features dominate predictions, with **item group** having the strongest positive effect.  
- Location and time (state, district, year, month) have little impact; item category slightly reduces predicted values.  
- Overall, the model relies mainly on **item characteristics**, suggesting that improving item grouping or classification would enhance accuracy, while adding temporal or geographic data provides limited benefit.  
""")

# --------------------
# INTERPRETATION
# --------------------
st.info("""
The Random Forest model outperforms other techniques in terms of accuracy and stability,
demonstrating its effectiveness in capturing seasonal and structural price patterns.
Residual and actual-versus-predicted analyses indicate minimal bias and strong generalization,
supporting its suitability for forecasting food prices in Pasar Mini markets.
""")
