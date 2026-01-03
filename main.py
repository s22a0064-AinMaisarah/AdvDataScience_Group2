import streamlit as st

st.set_page_config(page_title="Pasar Mini Dashboard")

# Import pages
page1 = st.Page("Descriptive.py", title="Descriptive Analysis", icon=":material/analytics:")
page2 = st.Page("Diagnostic.py", title="Diagnostic Analysis", icon=":material/scatter_plot:")
page3 = st.Page("Predictive.py", title="Predictive Analysis", icon=":material/radar:")
page3 = st.Page("Prescriptive.py", title="Prescriptive Analysis", icon=":material/radar:")

# Navigation
navigation = st.navigation(
    {
        "Pasar Mini Dashboard": [page1, page2, page3, page4]
    }
)

navigation.run()
