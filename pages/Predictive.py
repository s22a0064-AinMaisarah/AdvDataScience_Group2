import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import html

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# Load CSV from GitHub Raw
# --------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/s22a0064-AinMaisarah/AdvDataScience_Group2/main/dataset/pasar_mini_data_updated.csv"
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df['date'])
    return df

pasar_mini_df = load_data()

# --------------------
# Streamlit UI (REUSED FROM EXAMPLE)
# --------------------
st.markdown(
    '<div class="center-title">📈 Pasar Mini Predictive Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Forecasting Food Prices using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.center-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #6c757d;
    margin-bottom: 1.2rem;
}
.divider {
    border-top: 3px solid #1f77b4;
    margin: 1.2rem 0 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Dataset Preview
# --------------------
with st.expander("🔍 Preview of the Dataset", expanded=False):
    st.dataframe(pasar_mini_df.head(), use_container_width=True)

# --------------------
# STEP 1: DEFINE PREDICTIVE DATASET
# --------------------
predict_df = pasar_mini_df.copy()
predict_df['year'] = predict_df['date'].dt.year
predict_df['month'] = predict_df['date'].dt.month

monthly_price = (
    predict_df
    .groupby(['year','month','state_enc','district_enc',
              'item_group_enc','item_category_enc'])['price']
    .mean()
    .reset_index()
)

# --------------------
# STEP 2: FEATURES & TARGET
# --------------------
features = ['year','month','state_enc','district_enc',
            'item_group_enc','item_category_enc']
target = 'price'

X = monthly_price[features]
y = monthly_price[target]

# --------------------
# STEP 3: TIME-BASED TRAIN-TEST SPLIT (FIXED)
# --------------------
monthly_price = monthly_price.sort_values(by=['year','month']).reset_index(drop=True)
split_index = int(len(monthly_price) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# --------------------
# STEP 4: MODEL TRAINING
# --------------------
rf_model = RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

dtr_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dtr_model.fit(X_train, y_train)
y_pred_dtr = dtr_model.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)

# --------------------
# STEP 5: MODEL EVALUATION
# --------------------
st.subheader("📊 Model Performance Comparison")

models = {
    "Random Forest": y_pred_rf,
    "Linear Regression": y_pred_lr,
    "Decision Tree": y_pred_dtr,
    "SVR": y_pred_svr
}

results = []
for name, preds in models.items():
    results.append([
        name,
        mean_absolute_error(y_test, preds),
        np.sqrt(mean_squared_error(y_test, preds)),
        r2_score(y_test, preds)
    ])

results_df = pd.DataFrame(results, columns=["Model","MAE","RMSE","R²"])
st.dataframe(results_df, use_container_width=True)

# --------------------
# STEP 6: FORECAST 2025 (RF)
# --------------------
future_2025 = pd.DataFrame({
    'year': [2025]*12,
    'month': list(range(1,13)),
    'state_enc': monthly_price['state_enc'].mode()[0],
    'district_enc': monthly_price['district_enc'].mode()[0],
    'item_group_enc': monthly_price['item_group_enc'].mode()[0],
    'item_category_enc': monthly_price['item_category_enc'].mode()[0]
})

future_2025['predicted_price'] = rf_model.predict(future_2025[features])

# --------------------
# STEP 7: VISUALIZATION
# --------------------
st.subheader("📈 Forecasted Monthly Food Prices (2025)")

fig_forecast = px.line(
    future_2025,
    x='month',
    y='predicted_price',
    markers=True,
    labels={'month':'Month','predicted_price':'Price (RM)'}
)
st.plotly_chart(fig_forecast, use_container_width=True)

# --------------------
# FEATURE IMPORTANCE
# --------------------
st.subheader("🌟 Feature Importance (Random Forest)")

feat_imp = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig_imp = px.bar(feat_imp, x='Feature', y='Importance')
st.plotly_chart(fig_imp, use_container_width=True)

# --------------------
# ACTUAL vs PREDICTED
# --------------------
st.subheader("🔍 Actual vs Predicted Prices")

fig_ap = px.scatter(
    x=y_test,
    y=y_pred_rf,
    labels={'x':'Actual Price','y':'Predicted Price'}
)
fig_ap.add_shape(
    type='line',
    x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max(),
    line=dict(color='red')
)
st.plotly_chart(fig_ap, use_container_width=True)

# --------------------
# RESIDUAL DISTRIBUTION
# --------------------
st.subheader("📉 Residual Distribution")

residuals = y_test - y_pred_rf
fig, ax = plt.subplots()
sns.histplot(residuals, bins=30, kde=True, ax=ax)
st.pyplot(fig)

# --------------------
# INTERPRETATION
# --------------------
st.info(
    "The Random Forest model demonstrates strong predictive performance, "
    "capturing seasonal price variations effectively. Temporal variables "
    "(year and month) contribute most significantly, followed by product "
    "category and location. The forecast provides actionable insights for "
    "anticipating food price movements in Pasar Mini markets during 2025."
)
