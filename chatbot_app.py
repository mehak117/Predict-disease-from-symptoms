import streamlit as st
import pickle
import sqlite3
from datetime import datetime

# -----------------------------
# Load Model Files
# -----------------------------
model = pickle.load(open("disease_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
symptoms = pickle.load(open("symptoms.pkl", "rb"))

# -----------------------------
# Database Connection
# -----------------------------
conn = sqlite3.connect("healthcare.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patient_records (
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

conn.commit()

# -----------------------------
# Disease Precautions
# -----------------------------
disease_precautions = {

    "Fungal infection": [
        "Keep the affected area clean",
        "Keep skin dry",
        "Avoid sharing personal items",
        "Consult a dermatologist"
    ],

    "Allergy": [
        "Avoid allergens",
        "Stay hydrated",
        "Keep surroundings clean",
        "Take prescribed medicines"
    ],

    "Common Cold": [
        "Take adequate rest",
        "Drink warm fluids",
        "Stay hydrated",
        "Avoid cold exposure"
    ],

    "Diabetes": [
        "Monitor blood sugar regularly",
        "Follow a healthy diet",
        "Exercise daily",
        "Take medicines as prescribed"
    ],

    "Hypertension": [
        "Reduce salt intake",
        "Exercise regularly",
        "Manage stress",
        "Monitor blood pressure"
    ],

    "GERD": [
        "Avoid spicy foods",
        "Eat smaller meals",
        "Do not lie down after eating",
        "Maintain healthy body weight"
    ]
}

# -----------------------------
# Streamlit Page
# -----------------------------
st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 AI Healthcare Chatbot")
st.markdown("### Patient Information")

# -----------------------------
# Patient Details
# -----------------------------
name = st.text_input("Enter Patient Name")

age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=120,
    value=20
)

gender = st.selectbox(
    "Select Gender",
    ["Male", "Female", "Other"]
)

# -----------------------------
# Symptoms Selection
# -----------------------------
st.markdown("### Select Symptoms")

selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptoms
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Disease"):

    if name.strip() == "":
        st.warning("Please enter patient name.")

    elif len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:

        # Create Input Vector
        input_data = [0] * len(symptoms)

        for symptom in selected_symptoms:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                input_data[index] = 1

        # Prediction
        prediction = model.predict([input_data])

        disease = encoder.inverse_transform(prediction)[0]

        # Confidence Score
        probabilities = model.predict_proba([input_data])
        confidence = max(probabilities[0]) * 100

        # Save to Database
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

        # -----------------------------
        # Display Results
        # -----------------------------
        st.success(f"Hello {name}")

        st.markdown("## Patient Details")

        st.write("Name:", name)
        st.write("Age:", age)
        st.write("Gender:", gender)

        st.markdown("## Disease Prediction")

        st.success(f"Predicted Disease: {disease}")

        st.info(f"Prediction Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

        # -----------------------------
        # Precautions
        # -----------------------------
        st.markdown("## Recommended Precautions")

        if disease in disease_precautions:

            for precaution in disease_precautions[disease]:
                st.write(f"✔ {precaution}")

        else:

            st.write("✔ Consult a healthcare professional")
            st.write("✔ Take adequate rest")
            st.write("✔ Stay hydrated")
            st.write("✔ Follow medical advice")

        # Disclaimer
        st.warning(
            "This chatbot is for educational purposes only and does not replace professional medical advice."
        )

# -----------------------------
# View Patient History
# -----------------------------
st.markdown("---")

if st.button("Show Patient History"):

    records = cursor.execute(
        "SELECT * FROM patient_records ORDER BY id DESC"
    ).fetchall()

    if len(records) == 0:
        st.info("No records found.")

    else:

        for row in records:

            st.write("--------------------------------------------------")
            st.write("Patient ID:", row[0])
            st.write("Name:", row[1])
            st.write("Age:", row[2])
            st.write("Gender:", row[3])
            st.write("Symptoms:", row[4])
            st.write("Disease:", row[5])
            st.write(f"Confidence: {row[6]:.2f}%")
            st.write("Date:", row[7])
