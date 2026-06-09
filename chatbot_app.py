import streamlit as st
import pickle
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🏥",
    layout="wide"
)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("healthcare.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS patient_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    symptoms TEXT,
    disease TEXT,
    confidence REAL,
    prediction_date TEXT,
    created_by TEXT
)
""")

cursor.execute("""
INSERT OR IGNORE INTO users(name, username, password, role)
VALUES('Admin', 'admin', 'admin123', 'admin')
""")

conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# ---------------- LOGIN / REGISTER ----------------
if not st.session_state.logged_in:

    st.title("🏥 AI Healthcare Chatbot")
    st.subheader("Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Login As", ["user", "admin"])

        if st.button("Login"):
            cursor.execute("""
            SELECT * FROM users
            WHERE username=? AND password=? AND role=?
            """, (username, password, role))

            user = cursor.fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = role
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid username, password or role")

    with tab2:
        name = st.text_input("Full Name")
        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")

        if st.button("Register"):
            if name == "" or new_username == "" or new_password == "":
                st.warning("Please fill all fields")
            else:
                try:
                    cursor.execute("""
                    INSERT INTO users(name, username, password, role)
                    VALUES(?,?,?,?)
                    """, (name, new_username, new_password, "user"))

                    conn.commit()
                    st.success("Registration successful. Now login as user.")
                except:
                    st.error("Username already exists")

    st.stop()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("disease_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
symptoms = pickle.load(open("symptoms.pkl", "rb"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏥 Navigation")
st.sidebar.write("Logged in as:", st.session_state.username)
st.sidebar.write("Role:", st.session_state.role)

if st.session_state.role == "admin":
    menu = st.sidebar.radio(
        "Select Page",
        ["Admin Dashboard", "Disease Prediction", "Patient History", "Users"]
    )
else:
    menu = st.sidebar.radio(
        "Select Page",
        ["Disease Prediction", "My History"]
    )

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.rerun()

# ---------------- ADMIN DASHBOARD ----------------
if menu == "Admin Dashboard":

    st.title("📊 Admin Dashboard")

    df = pd.read_sql_query(
        "SELECT * FROM patient_records",
        conn
    )

    total_patients = len(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", total_patients)

    if not df.empty:
        col2.metric("Male Patients", len(df[df["gender"] == "Male"]))
        col3.metric("Female Patients", len(df[df["gender"] == "Female"]))

        st.subheader("Disease Distribution")
        st.bar_chart(df["disease"].value_counts())

        st.subheader("Recent Patient Records")
        st.dataframe(df.tail(10))
    else:
        st.info("No records found")

# ---------------- DISEASE PREDICTION ----------------
elif menu == "Disease Prediction":

    st.title("🏥 Disease Prediction")

    name = st.text_input("Patient Name")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    selected_symptoms = st.multiselect(
        "Select Symptoms",
        symptoms
    )

    if st.button("Predict Disease"):

        if name.strip() == "":
            st.warning("Enter patient name")

        elif len(selected_symptoms) == 0:
            st.warning("Select at least one symptom")

        else:
            input_data = [0] * len(symptoms)

            for symptom in selected_symptoms:
                index = symptoms.index(symptom)
                input_data[index] = 1

            prediction = model.predict([input_data])

            disease = encoder.inverse_transform(prediction)[0]

            confidence = max(
                model.predict_proba([input_data])[0]
            ) * 100

            cursor.execute("""
            INSERT INTO patient_records
            (name, age, gender, symptoms, disease,
            confidence, prediction_date, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                name,
                age,
                gender,
                ", ".join(selected_symptoms),
                disease,
                confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                st.session_state.username
            ))

            conn.commit()

            st.success(f"Predicted Disease: {disease}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.progress(int(confidence))

            st.subheader("Precautions")
            st.write("✔ Take adequate rest")
            st.write("✔ Stay hydrated")
            st.write("✔ Eat healthy food")
            st.write("✔ Consult a doctor if symptoms persist")

# ---------------- ADMIN PATIENT HISTORY ----------------
elif menu == "Patient History":

    st.title("📋 All Patient History")

    records = pd.read_sql_query(
        "SELECT * FROM patient_records ORDER BY id DESC",
        conn
    )

    if records.empty:
        st.info("No patient history found")
    else:
        st.dataframe(records)

        csv = records.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download All Records",
            csv,
            "patient_records.csv",
            "text/csv"
        )

# ---------------- USER HISTORY ----------------
elif menu == "My History":

    st.title("📋 My Prediction History")

    records = pd.read_sql_query(
        "SELECT * FROM patient_records WHERE created_by=? ORDER BY id DESC",
        conn,
        params=(st.session_state.username,)
    )

    if records.empty:
        st.info("No history found")
    else:
        st.dataframe(records)

# ---------------- USERS PAGE ----------------
elif menu == "Users":

    st.title("👥 Registered Users")

    users = pd.read_sql_query(
        "SELECT id, name, username, role FROM users",
        conn
    )

    st.dataframe(users)
