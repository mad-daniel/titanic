# pages/4_Interactive_Analysis.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Interactive Analysis", layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv("titanic_cleaned.csv") 

    if 'Cabin' in data.columns:
        data['Deck'] = data['Cabin'].str[0]
        data['Deck'] = data['Deck'].fillna('Unknown')
    else:
        data['Deck'] = 'Unknown'

    data['Family_Size'] = data['SibSp'] + data['Parch'] + 1
  
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    return data

df = load_data()

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>Interactive Analysis</h1>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("Filter Options")

col1, col2, col3 = st.columns(3)

with col1:
    sex_options = ["All"] + sorted(df['Sex'].dropna().unique().tolist())
    selected_sex = st.selectbox("Select Sex", options=sex_options)

with col2:
    pclass_options = ["All"] + sorted(df['Pclass'].dropna().unique().tolist())
    selected_pclass = st.selectbox("Select Passenger Class", options=pclass_options)

with col3:
    embarked_options = ["All"] + sorted(df['Embarked'].dropna().unique().tolist())
    selected_embarked = st.selectbox("Select Embarkation Point", options=embarked_options)

col4, col5, col6 = st.columns(3)

with col4:
    age_min = int(df['Age'].min()) if not df['Age'].isnull().all() else 0
    age_max = int(df['Age'].max()) if not df['Age'].isnull().all() else 100
    age_range = st.slider("Select Age Range", age_min, age_max, (age_min, age_max))

with col5:
    fare_min = float(df['Fare'].min()) if not df['Fare'].isnull().all() else 0.0
    fare_max = float(df['Fare'].max()) if not df['Fare'].isnull().all() else 500.0
    fare_range = st.slider("Select Fare Range", fare_min, fare_max, (fare_min, fare_max), step=1.0)

with col6:
    family_min = int(df['Family_Size'].min())
    family_max = int(df['Family_Size'].max())
    family_range = st.slider("Select Family Size", family_min, family_max, (family_min, family_max))

st.markdown("---")

filtered_df = df.copy()

if selected_sex != "All":
    filtered_df = filtered_df[filtered_df['Sex'] == selected_sex]

if selected_pclass != "All":
    filtered_df = filtered_df[filtered_df['Pclass'] == selected_pclass]

if selected_embarked != "All":
    filtered_df = filtered_df[filtered_df['Embarked'] == selected_embarked]

filtered_df = filtered_df[
    (filtered_df['Age'] >= age_range[0]) & 
    (filtered_df['Age'] <= age_range[1]) & 
    (filtered_df['Fare'] >= fare_range[0]) & 
    (filtered_df['Fare'] <= fare_range[1]) &
    (filtered_df['Family_Size'] >= family_range[0]) &
    (filtered_df['Family_Size'] <= family_range[1])
]

st.header("Filtered Data")
st.write(f"Number of Passengers after Filtering: {filtered_df.shape[0]}")
st.dataframe(filtered_df, use_container_width=True)

st.markdown("---")

st.header("Survival Rate Based on Filters")

plots = [
    ("Overall Survival Rate", lambda: create_survival_rate_plot(filtered_df)),
    ("Survival Rate by Sex", lambda: create_survival_by_sex_plot(filtered_df, selected_sex)),
    ("Survival Rate by Passenger Class", lambda: create_survival_by_pclass_plot(filtered_df, selected_pclass)),
    ("Age Distribution", lambda: create_age_distribution_plot(filtered_df)),
    ("Fare Distribution", lambda: create_fare_distribution_plot(filtered_df)),
    ("Family Size Distribution", lambda: create_family_size_distribution_plot(filtered_df)),
    ("Fare vs. Age", lambda: create_fare_vs_age_plot(filtered_df)),
    ("Age Boxplot by Survival", lambda: create_age_boxplot(filtered_df)),
    ("Age Violin Plot by Passenger Class", lambda: create_age_violinplot(filtered_df)),
    ("Survival Rate by Deck", lambda: create_survival_by_deck_plot(filtered_df)),
    ("Title Distribution", lambda: create_title_distribution_plot(filtered_df))
]

def create_survival_rate_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Survived', data=data, palette="viridis", ax=ax)
    ax.set_xlabel("Survived (0 = No, 1 = Yes)")
    ax.set_ylabel("Count")
    ax.set_title("Overall Survival Rate")
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
    plt.tight_layout()
    st.pyplot(fig)

def create_survival_by_sex_plot(data, selected_sex):
    if selected_sex == "All":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Sex', hue='Survived', data=data, palette="viridis", ax=ax)
        ax.set_title("Survival Rate by Sex")
        ax.set_xlabel("Sex")
        ax.set_ylabel("Count")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        plt.tight_layout()
        st.pyplot(fig)

def create_survival_by_pclass_plot(data, selected_pclass):
    if selected_pclass == "All":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Pclass', hue='Survived', data=data, palette="viridis", ax=ax)
        ax.set_title("Survival Rate by Passenger Class")
        ax.set_xlabel("Passenger Class")
        ax.set_ylabel("Count")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        plt.tight_layout()
        st.pyplot(fig)

def create_age_distribution_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data['Age'].dropna(), bins=30, kde=True, color="skyblue", ax=ax)
    ax.set_title("Age Distribution of Passengers")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig)

def create_fare_distribution_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Pclass', y='Fare', data=data, palette="viridis", ax=ax)
    ax.set_title("Fare Distribution by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Fare")
    plt.tight_layout()
    st.pyplot(fig)

def create_family_size_distribution_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='Family_Size', data=data, palette="viridis", ax=ax)
    ax.set_title("Family Size Distribution")
    ax.set_xlabel("Family Size")
    ax.set_ylabel("Count")
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
    plt.tight_layout()
    st.pyplot(fig)

def create_fare_vs_age_plot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Fare', hue='Survived', data=data, palette="viridis", ax=ax)
    ax.set_title("Fare vs. Age by Survival")
    ax.set_xlabel("Age")
    ax.set_ylabel("Fare")
    plt.tight_layout()
    st.pyplot(fig)

def create_age_boxplot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Survived', y='Age', data=data, palette="viridis", ax=ax)
    ax.set_title("Age Distribution by Survival")
    ax.set_xlabel("Survived")
    ax.set_ylabel("Age")
    plt.tight_layout()
    st.pyplot(fig)

def create_age_violinplot(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x='Pclass', y='Age', data=data, palette="viridis", ax=ax)
    ax.set_title("Age Distribution by Passenger Class")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Age")
    plt.tight_layout()
    st.pyplot(fig)

def create_survival_by_deck_plot(data):
    if 'Deck' in data.columns and data['Deck'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Deck', hue='Survived', data=data, palette="viridis", ax=ax)
        ax.set_title("Survival Rate by Deck")
        ax.set_xlabel("Deck")
        ax.set_ylabel("Count")
        ax.legend(title='Survived', labels=['No', 'Yes'])
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        plt.tight_layout()
        st.pyplot(fig)

def create_title_distribution_plot(data):
    if 'Title' in data.columns and data['Title'].nunique() > 1:
        title_counts = data['Title'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=title_counts.index, y=title_counts.values, palette="viridis", ax=ax)
        ax.set_title("Title Distribution")
        ax.set_xlabel("Title")
        ax.set_ylabel("Count")
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, int(height), ha="center")
        plt.tight_layout()
        st.pyplot(fig)

plot_list = [
    ("Overall Survival Rate", lambda: create_survival_rate_plot(filtered_df)),
    ("Survival Rate by Sex", lambda: create_survival_by_sex_plot(filtered_df, selected_sex)),
    ("Survival Rate by Passenger Class", lambda: create_survival_by_pclass_plot(filtered_df, selected_pclass)),
    ("Age Distribution", lambda: create_age_distribution_plot(filtered_df)),
    ("Fare Distribution", lambda: create_fare_distribution_plot(filtered_df)),
    ("Family Size Distribution", lambda: create_family_size_distribution_plot(filtered_df)),
    ("Fare vs. Age", lambda: create_fare_vs_age_plot(filtered_df)),
    ("Age Boxplot by Survival", lambda: create_age_boxplot(filtered_df)),
    ("Age Violin Plot by Passenger Class", lambda: create_age_violinplot(filtered_df)),
    ("Survival Rate by Deck", lambda: create_survival_by_deck_plot(filtered_df)),
    ("Title Distribution", lambda: create_title_distribution_plot(filtered_df))
]

for i in range(0, len(plot_list), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(plot_list):
            title, plot_func = plot_list[i + j]
            with cols[j]:
                st.subheader(title)
                plot_func()

st.markdown("---")

st.header("Download Filtered Data")
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(filtered_df)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name='filtered_titanic_data.csv',
    mime='text/csv',
)
