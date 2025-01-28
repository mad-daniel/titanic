# pages/2_Data_Overview.py

import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt 

st.set_page_config(page_title="Data Overview", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("titanic_cleaned.csv")
    return data

df = load_data()

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Data Overview</h1>
    """, unsafe_allow_html=True)

st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Passengers", f"{df.shape[0]}")

with col2:
    st.metric("Total Features", f"{df.shape[1]}")

with col3:
    st.metric("Missing Values", f"{df.isnull().sum().sum()}")

with col4:
    if 'Survived' in df.columns:
        survival_rate = (df['Survived'].mean() * 100)
        st.metric("Survival Rate", f"{survival_rate:.2f}%")
    else:
        st.metric("Survival Rate", "N/A")

st.markdown("---")

st.subheader("First Five Rows of the Dataset")
st.dataframe(df.head(), height=200)

st.markdown("---")

st.subheader("Dataset Information")

buffer = io.StringIO()
df.info(buf=buffer)
info = buffer.getvalue()

st.code(info, language='python')

st.markdown("---")

st.subheader("Statistical Summary")

styled_summary = (
    df.describe()
      .style
      .background_gradient(cmap='Blues')
      .format("{:.2f}")
      .set_table_styles([
          {
              'selector': 'th',
              'props': [
                  ('background-color', '#f7f7f9'),
                  ('color', '#333'),
                  ('font-weight', 'bold'),
                  ('text-align', 'center')
              ]
          },
          {
              'selector': 'td',
              'props': [
                  ('padding', '8px'),
                  ('text-align', 'center')
              ]
          },
      ])
      .hide(axis="index")
)

st.dataframe(styled_summary, use_container_width=True, height=300)

st.markdown("---")

st.subheader("Missing Values")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_df = missing_values.reset_index()
missing_df.columns = ['Feature', 'Missing Values']

st.table(missing_df.style.highlight_max(color='red', axis=0))

st.markdown("**Missing Values by Feature**")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Missing Values', y='Feature', data=missing_df, palette="Reds_d", ax=ax)
ax.set_title("Missing Values per Feature")
ax.set_xlabel("Number of Missing Values")
ax.set_ylabel("Feature")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.subheader("Unique Values per Feature")

categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

if categorical_features:
    unique_values = df[categorical_features].nunique().reset_index()
    unique_values.columns = ['Feature', 'Unique Values']
    
    st.table(unique_values.style.highlight_max(color='green', axis=0))
    
    st.markdown("**Unique Values by Feature**")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Unique Values', y='Feature', data=unique_values, palette="Greens_d", ax=ax2)
    ax2.set_title("Unique Values per Categorical Feature")
    ax2.set_xlabel("Number of Unique Values")
    ax2.set_ylabel("Feature")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.write("No categorical features found in the dataset.")

st.markdown("---")

st.subheader("Data Types Distribution")

data_types = df.dtypes.value_counts().reset_index()
data_types.columns = ['Data Type', 'Count']

st.table(data_types.style.highlight_max(color='purple', axis=0))

st.markdown("**Data Types Distribution**")
fig3, ax3 = plt.subplots()
ax3.pie(data_types['Count'], labels=data_types['Data Type'], autopct='%1.1f%%', colors=sns.color_palette("pastel"))
ax3.set_title("Distribution of Data Types")
plt.tight_layout()
st.pyplot(fig3)

st.markdown("---")
