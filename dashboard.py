import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="📊")

st.title("📊 Healthcare Dashboard")

conn = sqlite3.connect("healthcare.db")

df = pd.read_sql_query(
    "SELECT * FROM patient_records",
    conn
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Patients", len(df))

with col2:
    if len(df) > 0:
        st.metric(
            "Male Patients",
            len(df[df["gender"] == "Male"])
        )

with col3:
    if len(df) > 0:
        st.metric(
            "Female Patients",
            len(df[df["gender"] == "Female"])
        )

st.subheader("Recent Patient Records")

if len(df) > 0:
    st.dataframe(df)

    st.subheader("Disease Distribution")

    disease_counts = df["disease"].value_counts()

    st.bar_chart(disease_counts)

else:
    st.info("No patient records found.")
