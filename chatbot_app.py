import os
import pickle
import streamlit as st

# 1. SET UP ABSOLUTE PATHING TO LOAD THE MODEL
# This ensures Streamlit Cloud can find 'disease_model.pkl' regardless of the working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "disease_model.pkl")

@st.cache_resource
def load_model():
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(
            f"**Error:** Could not find `{os.path.basename(model_path)}` in your repository root. "
            "Please ensure the file is uploaded to GitHub and named correctly."
        )
        return None

model = load_model()

# 2. APP USER INTERFACE (UI)
st.set_page_config(page_title="Disease Predictor Chatbot", page_icon="🩺", layout="centered")

st.title("🩺 Symptom-Based Disease Predictor")
st.write("Welcome! Please describe or select your symptoms below to get a preliminary prediction.")

st.divider()

# Only run the app logic if the model successfully loaded
if model is not None:
    # --- CHAT / INPUT INTERFACE ---
    # Note: Depending on how your machine learning model was trained, it expects specific features.
    # Replace the list below with the exact symptom features your model handles.
    available_symptoms = [
        "Fever", "Cough", "Fatigue", "Body Ache", "Headache", 
        "Nausea", "Sore Throat", "Shortness of Breath", "Skin Rash"
    ]
    
    st.subheader("Select Your Symptoms")
    selected_symptoms = st.multiselect(
        "Choose all symptoms you are currently experiencing:",
        options=sorted(available_symptoms)
    )
    
    # --- PREDICTION LOGIC ---
    if st.button("Predict Outcome", type="primary"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom to analyze.")
        else:
            with st.spinner("Analyzing symptoms against our model..."):
                # TODO: Convert selected_symptoms into the exact input format your model expects.
                # For example, if your model takes a binary array (1 for present, 0 for absent):
                # input_data = [1 if symptom in selected_symptoms else 0 for symptom in available_symptoms]
                
                try:
                    # Dummy placeholder vector for demonstration—replace with your actual input vector parsing:
                    # prediction = model.predict([input_data])[0]
                    
                    # Temporarily using a mock fallback if your feature shape isn't mapped yet:
                    prediction = "Sample Disease (Please map your input features in the code)"
                    
                    st.success("### Analysis Complete")
                    st.markdown(f"Based on the reported symptoms, the model predicts: **{prediction}**")
                    
                    st.info(
                        "⚠️ **Disclaimer:** This is an AI-generated assessment based on historical data patterns. "
                        "It does not substitute for professional medical advice, diagnosis, or treatment."
                    )
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.info("Ensure the input format matches what your model was trained on.")
