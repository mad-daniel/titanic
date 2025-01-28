# titanic_eda.py

import streamlit as st

st.set_page_config(page_title="Titanic EDA", layout="wide")

st.sidebar.title("Titanic EDA Navigation")
st.sidebar.markdown("""
Navigate through the pages using the sidebar:
- **Home**: Introduction and overview.
- **Data Overview**: Basic dataset insights.
- **Data Visualization**: Various charts and graphs.
- **Interactive Analysis**: Apply filters and explore.
- **Data Cleaning**: Preprocess the data.
""")

st.title("Welcome to the Titanic EDA App")
st.markdown("""
This application provides an interactive exploratory data analysis of the Titanic dataset. Use the sidebar to navigate through different sections and gain insights into the factors that influenced survival rates.
""")

st.markdown("Daniel Kosbab")