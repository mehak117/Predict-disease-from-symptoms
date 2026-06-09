import streamlit as st
import pickle
import sqlite3
import pandas as pd
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🏥",
    layout="wide"
)

# =========================
# LOAD MODEL FILES
# =========================
model = pickle.load(open("disease_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
symptoms = pickle.load(open("symptoms.pkl", "rb"))

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("healthcare.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patient_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER,
    gender TEXT,
    symptoms TEXT,
    disease TEXT,
    confidence REAL,
    prediction_date TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

conn.commit()

# =========================
# DEFAULT ADMIN
# =========================
cursor.execute("SELECT * FROM users WHERE username='admin'")
admin_exists = cursor.fetchone()

if not admin_exists:
    cursor.execute(
        "INSERT INTO users(username, password, role) VALUES (?, ?, ?)",
        ("admin", "admin123", "admin")
    )
    conn.commit()

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = ""

# =========================
# LOGIN / REGISTER PAGE
# =========================
if not st.session_state.logged_in:

    st.title("🏥 AI Healthcare Chatbot Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, password)
            )
            user = cursor.fetchone()

            if user:
                st.session_state.logged_in = True
                st.session_state.username = user[1]
                st.session_state.role = user[3]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

    with tab2:
        st.subheader("User Registration")

        new_username = st.text_input("Create Username")
        new_password = st.text_input("Create Password", type="password")

        if st.button("Register"):
            if new_username.strip() == "" or new_password.strip() == "":
                st.warning("Please fill all fields.")
            else:
                try:
                    cursor.execute(
                        "INSERT INTO users(username, password, role) VALUES (?, ?, ?)",
                        (new_username, new_password, "user")
                    )
                    conn.commit()
                    st.success("Registration successful. Please login.")
                except sqlite3.IntegrityError:
                    st.error("Username already exists.")

# =========================
# MAIN APP
# =========================
else:

    st.sidebar.title("🏥 Navigation")
    st.sidebar.write(f"👤 Logged in as: **{st.session_state.username}**")
    st.sidebar.write(f"🔐 Role: **{st.session_state.role}**")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()

    if st.session_state.role == "admin":
        menu = st.sidebar.radio(
            "Select Page",
            ["Dashboard", "Disease Prediction", "Patient History", "User Management"]
        )
    else:
        menu = st.sidebar.radio(
            "Select Page",
            ["Dashboard", "Disease Prediction", "Patient History"]
        )

    # =========================
    # DASHBOARD
    # =========================
    if menu == "Dashboard":

        st.title("📊 Healthcare Dashboard")

        df = pd.read_sql_query("SELECT * FROM patient_records", conn)

        total_patients = len(df)
        male_patients = 0
        female_patients = 0

        if not df.empty:
            male_patients = len(df[df["gender"] == "Male"])
            female_patients = len(df[df["gender"] == "Female"])

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Patients", total_patients)
        col2.metric("Male Patients", male_patients)
        col3.metric("Female Patients", female_patients)

        if not df.empty:
            st.subheader("Disease Distribution")
            disease_count = df["disease"].value_counts()
            st.bar_chart(disease_count)

            st.subheader("Recent Records")
            st.dataframe(df.tail(10))
        else:
            st.info("No patient records found.")

    # =========================
    # DISEASE PREDICTION
    # =========================
    elif menu == "Disease Prediction":

        st.title("👨‍⚕️ Patient Symptom Evaluation")

        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Patient Name")

        with col2:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=20
            )

        with col3:
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )

        st.write("Search and select your symptoms:")

        selected_symptoms = st.multiselect(
            "",
            symptoms,
            placeholder="Choose options"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Analyze System Mapping"):

            if name.strip() == "":
                st.warning("Enter patient name.")

            elif len(selected_symptoms) == 0:
                st.warning("Select symptoms.")

            else:
                input_data = [0] * len(symptoms)

                for symptom in selected_symptoms:
                    index = symptoms.index(symptom)
                    input_data[index] = 1

                prediction = model.predict([input_data])

                disease = encoder.inverse_transform(prediction)[0]

                confidence = max(model.predict_proba([input_data])[0]) * 100

                cursor.execute("""
                INSERT INTO patient_records
                (name, age, gender, symptoms, disease, confidence, prediction_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    age,
                    gender,
                    ", ".join(selected_symptoms),
                    disease,
                    confidence,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))

                conn.commit()

                st.success("Analysis Completed Successfully")

                st.subheader("Patient Details")

                r1, r2, r3 = st.columns(3)

                r1.info(f"👤 Name: {name}")
                r2.info(f"🎂 Age: {age}")
                r3.info(f"⚧ Gender: {gender}")

                st.subheader("Prediction Result")

                st.success(f"🩺 Predicted Disease: {disease}")
                st.info(f"📈 Confidence: {confidence:.2f}%")

                st.progress(int(confidence))

                st.subheader("Precautions")

                precautions = [
                    "Take adequate rest",
                    "Stay hydrated",
                    "Eat healthy food",
                    "Consult a doctor if symptoms persist"
                ]

                for p in precautions:
                    st.write("✔", p)

    # =========================
    # PATIENT HISTORY
    # =========================
    elif menu == "Patient History":

        st.title("📋 Patient History")

        records = pd.read_sql_query(
            "SELECT * FROM patient_records ORDER BY id DESC",
            conn
        )

        if records.empty:
            st.info("No patient history found.")
        else:
            st.dataframe(records)

            csv = records.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Records",
                csv,
                "patient_records.csv",
                "text/csv"
            )

    # =========================
    # USER MANAGEMENT ADMIN ONLY
    # =========================
    elif menu == "User Management":

        if st.session_state.role != "admin":
            st.error("Access denied. Admin only.")
        else:
            st.title("👨‍💼 User Management")

            users_df = pd.read_sql_query(
                "SELECT id, username, role FROM users",
                conn
            )

            st.dataframe(users_df)
