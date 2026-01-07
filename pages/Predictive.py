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

future_2025 = pd.DataFrame({
    "year": [2025]*12,
    "month": list(range(1,13)),
    "state_enc": monthly_price["state_enc"].mode()[0],
    "district_enc": monthly_price["district_enc"].mode()[0],
    "item_group_enc": monthly_price["item_group_enc"].mode()[0],
    "item_category_enc": monthly_price["item_category_enc"].mode()[0],
})

future_2025["predicted_price"] = rf_model.predict(future_2025[features])

# --------------------
# Forecasted Line Chart (Trend Over Time)
# --------------------

with st.expander("📉 Forecast Trend: Monthly Food Prices (2025)", expanded=True):

    st.subheader("📉 Forecasted Price Trend (Line Chart)")

    fig_line = px.line(
        future_2025,
        x="month",
        y="predicted_price",
        markers=True,
        title="Forecasted Monthly Food Prices in Pasar Mini (2025)",
        labels={
            "month": "Month",
            "predicted_price": "Predicted Price (RM)"
        }
    )

    st.plotly_chart(fig_line, use_container_width=True)

    st.caption(
        "📌 This line chart illustrates the overall trend and seasonal movement of "
        "forecasted food prices in Pasar Mini markets for the year 2025."
    )

# --------------------
# Monthly Forecast Bar Chart (Month to Month Comparison)
# --------------------
with st.expander("📊 Monthly Price Comparison (Bar Chart)", expanded=False):

    st.subheader("📊 Monthly Forecast Comparison")

    fig_bar = px.bar(
        future_2025,
        x="month",
        y="predicted_price",
        text="predicted_price",
        title="Predicted Food Prices by Month in Pasar Mini (2025)",
        labels={
            "month": "Month",
            "predicted_price": "Predicted Price (RM)"
        }
    )

    fig_bar.update_traces(
        texttemplate="RM %{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption(
        "📌 This bar chart provides a clear month-to-month comparison of predicted "
        "food prices, allowing users to identify higher or lower pricing periods."
    )

# --------------------
# ACTUAL VS PREDICTED
# --------------------
st.subheader("2. 🔍 Actual vs Predicted Comparison")

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

# --------------------
# INTERPRETATION
# --------------------
st.info("""
The Random Forest model outperforms other techniques in terms of accuracy and stability,
demonstrating its effectiveness in capturing seasonal and structural price patterns.
Residual and actual-versus-predicted analyses indicate minimal bias and strong generalization,
supporting its suitability for forecasting food prices in Pasar Mini markets.
""")
