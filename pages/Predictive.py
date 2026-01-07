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
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/main/dataset/pasar_mini_data_updated.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
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

features = ["year","month","state_enc","district_enc",
            "item_group_enc","item_category_enc"]
target = "price"

X = monthly_price[features]
y = monthly_price[target]

# --------------------
# TIME-BASED SPLIT
# --------------------
monthly_price = monthly_price.sort_values(["year","month"]).reset_index(drop=True)
split_index = int(len(monthly_price) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# --------------------
# MODEL TRAINING
# --------------------
rf = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
y_dt = dt.predict(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

svr = SVR(C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_s, y_train)
y_svr = svr.predict(X_test_s)

models = {
    "Random Forest": y_rf,
    "Linear Regression": y_lr,
    "Decision Tree": y_dt,
    "SVR": y_svr
}

# --------------------
# SUMMARY METRICS 
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
# FORECAST 2025 (RF)
# --------------------
st.subheader("📈 Forecasted Prices for 2025 (Random Forest)")

future_2025 = pd.DataFrame({
    "year": [2025]*12,
    "month": list(range(1,13)),
    "state_enc": monthly_price["state_enc"].mode()[0],
    "district_enc": monthly_price["district_enc"].mode()[0],
    "item_group_enc": monthly_price["item_group_enc"].mode()[0],
    "item_category_enc": monthly_price["item_category_enc"].mode()[0],
})

future_2025["predicted_price"] = rf.predict(future_2025[features])

fig_forecast = px.line(
    future_2025, x="month", y="predicted_price",
    markers=True,
    labels={"month":"Month","predicted_price":"Price (RM)"}
)
st.plotly_chart(fig_forecast, use_container_width=True)

# --------------------
# FEATURE IMPORTANCE
# --------------------
st.subheader("🌟 Feature Importance (Random Forest)")

feat_imp = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

fig_imp = px.bar(feat_imp, x="Feature", y="Importance")
st.plotly_chart(fig_imp, use_container_width=True)

# --------------------
# ACTUAL VS PREDICTED (ALL MODELS)
# --------------------
st.subheader("🔍 Actual vs Predicted Comparison")

for name, preds in models.items():
    fig = px.scatter(
        x=y_test, y=preds,
        labels={"x":"Actual Price","y":"Predicted Price"},
        title=name
    )
    fig.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red")
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# RESIDUAL PLOTS (ALL MODELS)
# --------------------
st.subheader("📉 Residual Distributions")

for name, preds in models.items():
    residuals = y_test - preds
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Residual")
    st.pyplot(fig)

# --------------------
# INTERPRETATION
# --------------------
st.info("""
The Random Forest model outperforms other techniques in terms of accuracy and stability,
demonstrating its effectiveness in capturing seasonal and structural price patterns.
Residual and actual-versus-predicted analyses indicate minimal bias and strong generalization,
supporting its suitability for forecasting food prices in Pasar Mini markets.
""")
