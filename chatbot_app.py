import streamlit as st
import pickle

# 1. This MUST be the very first Streamlit command in your code
st.set_page_config(page_title="AI Healthcare Chatbot", page_icon="🏥")

# 2. Add an explicit error handler to give you clear feedback if files fail to load
try:
    model = pickle.load(open("disease_model.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
    symptoms = pickle.load(open("symptoms.pkl", "rb"))
except Exception as e:
    st.error(f"Failed to load model files. Error: {e}")
    st.info("Tip: If you see 'UnpicklingError', ensure your Streamlit Python version matches your local training environment.")
    st.stop()

st.title("🏥 AI Healthcare Chatbot")
st.write("Select your symptoms and predict possible disease.")

selected_symptoms = st.multiselect(
    "Choose Symptoms",
    symptoms
)

if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")

    else:
        input_data = [0] * len(symptoms)

        for symptom in selected_symptoms:
            input_data[symptoms.index(symptom)] = 1

        prediction = model.predict([input_data])
        disease = encoder.inverse_transform(prediction)[0]

        # Confidence
        probabilities = model.predict_proba([input_data])
        confidence = max(probabilities[0]) * 100

        st.success(f"Predicted Disease: {disease}")
        st.info(f"Prediction Confidence: {confidence:.2f}%")
        st.progress(int(confidence))

        # Sample precautions
        st.subheader("Precautions")

        precautions = [
            "Drink plenty of water",
            "Take adequate rest",
            "Maintain a healthy diet",
            "Consult a doctor if symptoms persist"
        ]

        for p in precautions:
            st.write("✔", p)

        st.warning(
            "This chatbot is for educational purposes only and does not replace professional medical advice."
        )
