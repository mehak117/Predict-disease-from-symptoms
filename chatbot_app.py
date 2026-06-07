import streamlit as st
import pickle

# Load model files
# Load model files
# NEW CORRECTED CODE
with open("disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("symptoms.pkl", "rb") as f:
    symptoms = pickle.load(f)

# Disease-specific precautions
disease_precautions = {
    "Fungal infection": [
        "Keep the affected area clean",
        "Keep skin dry",
        "Avoid sharing personal items",
        "Consult a dermatologist if needed"
    ],

    "Allergy": [
        "Avoid allergens",
        "Stay hydrated",
        "Keep surroundings clean",
        "Take prescribed medications"
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
        "Take medications as prescribed"
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
        "Maintain a healthy weight"
    ]
}

# Page settings
st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🏥"
)

# Title
st.title("🏥 AI Healthcare Chatbot")
st.write("Select your symptoms and predict possible disease.")

# Symptom selection
selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptoms
)

# Prediction button
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:

        # Create feature vector
        input_data = [0] * len(symptoms)

        for symptom in selected_symptoms:
            input_data[symptoms.index(symptom)] = 1

        # Predict disease
        prediction = model.predict([input_data])

        disease = encoder.inverse_transform(prediction)[0]

        # Confidence score
        probabilities = model.predict_proba([input_data])
        confidence = max(probabilities[0]) * 100

        # Display result
        st.success(f"🩺 Predicted Disease: {disease}")

        st.info(f"📊 Prediction Confidence: {confidence:.2f}%")

        st.progress(int(confidence))

        # Precautions
        st.subheader("🛡️ Recommended Precautions")

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
            "⚠️ This chatbot is for educational purposes only and does not replace professional medical advice."
        )
