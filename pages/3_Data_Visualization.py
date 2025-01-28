# pages/3_Data_Visualization.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Visualization", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("titanic_cleaned.csv")
    return data

df = load_data()

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Data Visualization</h1>
    """, unsafe_allow_html=True)

st.markdown("---")

st.sidebar.header("Visualization Settings")

plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    ("Survival Rate", "Survival by Sex", "Age Distribution", "Survival by Passenger Class", "Correlation Matrix", "Fare Distribution", "Family Size Distribution")
)

def create_plot(plot_type):
    if plot_type == "Survival Rate":
        st.subheader("Survival Rate")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Survived', data=df, palette="viridis", ax=ax)
        ax.set_xlabel("Survived (0 = No, 1 = Yes)")
        ax.set_ylabel("Count")
        ax.set_title("Overall Survival Rate")
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        st.pyplot(fig)

    elif plot_type == "Survival by Sex":
        st.subheader("Survival by Sex")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Sex', hue='Survived', data=df, palette="viridis", ax=ax)
        ax.set_title("Survival Rate by Sex")
        ax.set_xlabel("Sex")
        ax.set_ylabel("Count")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        st.pyplot(fig)

    elif plot_type == "Age Distribution":
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df['Age'].dropna(), bins=30, kde=True, color="skyblue", ax=ax)
        ax.set_title("Age Distribution of Passengers")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    elif plot_type == "Survival by Passenger Class":
        st.subheader("Survival by Passenger Class")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Pclass', hue='Survived', data=df, palette="viridis", ax=ax)
        ax.set_title("Survival Rate by Passenger Class")
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Count")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        st.pyplot(fig)

    elif plot_type == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
        ax.set_title("Correlation Matrix of Features")
        st.pyplot(fig)

    elif plot_type == "Fare Distribution":
        st.subheader("Fare Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Pclass', y='Fare', data=df, palette="viridis", ax=ax)
        ax.set_title("Fare Distribution by Passenger Class")
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Fare")
        st.pyplot(fig)

    elif plot_type == "Family Size Distribution":
        st.subheader("Family Size Distribution")
        # Create a new feature for family size
        df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Family_Size', data=df, palette="viridis", ax=ax)
        ax.set_title("Family Size Distribution")
        ax.set_xlabel("Family Size")
        ax.set_ylabel("Count")
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        st.pyplot(fig)

create_plot(plot_type)

st.markdown("---")

st.subheader("Additional Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Fare vs. Age**")
    fig6, ax6 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='viridis', ax=ax6)
    ax6.set_title("Fare vs. Age by Survival")
    ax6.set_xlabel("Age")
    ax6.set_ylabel("Fare")
    st.pyplot(fig6)

with col2:
    st.markdown("**Embarkation Point and Survival**")
    fig7, ax7 = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Embarked', hue='Survived', data=df, palette="viridis", ax=ax7)
    ax7.set_title("Survival Rate by Embarkation Point")
    ax7.set_xlabel("Embarkation Point")
    ax7.set_ylabel("Count")
    ax7.legend(title='Survived', labels=['No', 'Yes'])
    for p in ax7.patches:
        height = p.get_height()
        ax7.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
    st.pyplot(fig7)
