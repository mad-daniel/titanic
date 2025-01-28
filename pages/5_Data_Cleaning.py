# pages/5_Data_Cleaning.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Cleaning and Transformation", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    return data

df_original = load_data()
df = df_original.copy()

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Data Cleaning and Transformation</h1>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("Missing Values Before Cleaning")

fig_missing_before, ax_missing_before = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax_missing_before)
ax_missing_before.set_title("Missing Values Heatmap - Before Cleaning")
st.pyplot(fig_missing_before)

missing_before = df.isnull().sum()
missing_before = missing_before[missing_before > 0]
fig_bar_before, ax_bar_before = plt.subplots(figsize=(10, 4))
missing_before.sort_values(ascending=False).plot.bar(ax=ax_bar_before, color='skyblue')
ax_bar_before.set_title("Missing Values Count - Before Cleaning")
ax_bar_before.set_xlabel("Columns")
ax_bar_before.set_ylabel("Number of Missing Values")
st.pyplot(fig_bar_before)

st.markdown("---")

st.subheader("Handling Missing Values")

with st.expander("View Missing Values Handling Instructions"):
    st.write("""
        **Steps Taken to Handle Missing Values:**
        1. **Age:** Filled missing values with the median age.
        2. **Embarked:** Filled missing values with the mode (most frequent value).
    """)

age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
st.success(f"Filled missing 'Age' values with median: {age_median}")

embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
st.success(f"Filled missing 'Embarked' values with mode: {embarked_mode}")

st.markdown("---")

st.subheader("Missing Values After Cleaning")

fig_missing_after, ax_missing_after = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax_missing_after)
ax_missing_after.set_title("Missing Values Heatmap - After Cleaning")
st.pyplot(fig_missing_after)

missing_after = df.isnull().sum()
missing_after = missing_after[missing_after > 0]
if not missing_after.empty:
    fig_bar_after, ax_bar_after = plt.subplots(figsize=(10, 4))
    missing_after.sort_values(ascending=False).plot.bar(ax=ax_bar_after, color='salmon')
    ax_bar_after.set_title("Missing Values Count - After Cleaning")
    ax_bar_after.set_xlabel("Columns")
    ax_bar_after.set_ylabel("Number of Missing Values")
    st.pyplot(fig_bar_after)
else:
    st.write("No missing values remaining after cleaning.")

st.markdown("---")

st.subheader("Encoding Categorical Variables")

with st.expander("View Encoding Instructions"):
    st.write("""
        **Steps Taken to Encode Categorical Variables:**
        1. **Sex:** Encoded as male=0, female=1.
        2. **Embarked:** Encoded as C=0, Q=1, S=2.
    """)

df['Sex_Code'] = df['Sex'].map({'male': 0, 'female': 1})

df['Embarked_Code'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

st.subheader("Encoded Columns")
st.write(df[['Sex', 'Sex_Code', 'Embarked', 'Embarked_Code']].head())

st.markdown("---")

st.subheader("Feature Engineering")

with st.expander("View Feature Engineering Instructions"):
    st.write("""
        **Feature Created:**
        - **Family_Size:** Calculated as the sum of SibSp (siblings/spouses aboard) and Parch (parents/children aboard) plus one (the passenger themselves).
    """)

df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
st.success("Added 'Family_Size' feature (SibSp + Parch + 1):")
st.write(df[['SibSp', 'Parch', 'Family_Size']].head())

st.markdown("---")

st.subheader("Data Cleaning Summary")

col7, col8 = st.columns(2)

with col7:
    st.markdown("**Before Cleaning**")
    st.write(df_original.head())

with col8:
    st.markdown("**After Cleaning**")
    st.write(df.head())

st.markdown("---")

st.subheader("Saving the Cleaned Data")

if st.button("Save Cleaned Data"):
    df.to_csv("titanic_cleaned.csv", index=False)
    st.success("Cleaned data has been saved as 'titanic_cleaned.csv'.")

st.markdown("### Download Cleaned Data")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_cleaned = convert_df(df)

st.download_button(
    label="Download Cleaned CSV",
    data=csv_cleaned,
    file_name='titanic_cleaned.csv',
    mime='text/csv',
)

st.markdown("---")
st.subheader("Cleaned Data Preview")
st.dataframe(df.head(), use_container_width=True)
