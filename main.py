import streamlit as st

st.set_page_config(
    page_title="Pasar Mini Dashboard",
    layout="wide"
)

st.title("🏪 Pasar Mini Analytics Dashboard")

st.markdown("### 📌 Navigate to Analysis Pages")

st.page_link("pages/1_Descriptive.py", label="📊 Descriptive Analysis")
st.page_link("pages/2_Diagnostic.py", label="🔍 Diagnostic Analysis")
st.page_link("pages/3_Predictive.py", label="📈 Predictive Analysis")
st.page_link("pages/4_Prescriptive.py", label="🧠 Prescriptive Analysis")
