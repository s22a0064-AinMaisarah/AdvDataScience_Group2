# Main.py
import streamlit as st
import Descriptive
import Diagnostic
import Predictive
import Prescriptive

st.set_page_config(
    page_title="Pasar Mini Dashboard",
    layout="wide"
)

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
)

if page == "Descriptive":
    Descriptive.app()

elif page == "Diagnostic":
    Diagnostic.app()

elif page == "Predictive":
    Predictive.app()

elif page == "Prescriptive":
    Prescriptive.app()
