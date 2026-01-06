import streamlit as st

st.set_page_config(page_title="Pasar Mini Dashboard")

# Import pages (note the 'pages/' prefix)
page1 = st.Page("pages/Descriptive.py", title="Descriptive Analysis", icon=":material/analytics:")
page2 = st.Page("pages/Diagnostic.py", title="Diagnostic Analysis", icon=":material/scatter_plot:")
page3 = st.Page("pages/Predictive.py", title="Predictive Analysis", icon=":material/radar:")
page4 = st.Page("pages/Prescriptive.py", title="Prescriptive Analysis", icon=":material/radar:")

navigation = st.navigation(
    {
        "Pasar Mini Dashboard": [page1, page2, page3, page4]
    }
)

navigation.run()
run()
