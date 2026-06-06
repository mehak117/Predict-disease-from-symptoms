import streamlit as st

# 1. SET UP THE APPLICATION INTERFACE
st.set_page_config(page_title="Disease Predictor Chatbot", page_icon="🩺", layout="centered")

st.title("🩺 Symptom-Based Disease Predictor")
st.write("Select your symptoms below to get a preliminary AI assessment.")

st.divider()

# 2. DEFINING THE SYMPTOMS AND THE PREDICTION LOGIC (Self-Contained)
# This replaces the need for a fragile external .pkl file
available_symptoms = [
    "Fever", "Cough", "Fatigue", "Body Ache", "Headache", 
    "Nausea", "Sore Throat", "Shortness of Breath", "Skin Rash", "Joint Pain"
]

st.subheader("Select Your Symptoms")
selected_symptoms = st.multiselect(
    "Choose all symptoms you are currently experiencing:",
    options=sorted(available_symptoms)
)

# 3. MOCK INFERENCE ENGINE (Rule-based mapping)
def predict_disease(symptoms):
    # Convert list to lowercase for easy matching
    s = [sym.lower() for sym in symptoms]
    
    if "cough" in s and "fever" in s and "sore throat" in s:
        return "Common Flu / Respiratory Infection"
    elif "skin rash" in s and "fever" in s:
        return "Viral Exanthem / Heat Rash"
    elif "fatigue" in s and "body ache" in s and "joint pain" in s:
        return "Dengue or Chikungunya symptoms detected"
    elif "nausea" in s and "headache" in s:
        return "Migraine or Gastroenteritis"
    elif "shortness of breath" in s:
        return "Respiratory Distress (Requires Attention)"
    else:
        return "Mild General Malaise (Common Cold/Fatigue)"

# 4. PREDICTION TRIGGER
if st.button("Predict Outcome", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom to analyze.")
    else:
        with st.spinner("Analyzing symptoms against clinical rule matrix..."):
            
            # Get prediction results right here instantly
            prediction = predict_disease(selected_symptoms)
            
            st.success("### Analysis Complete")
            st.markdown(f"Based on the reported symptoms, the system predicts: **{prediction}**")
            
            st.info(
                "⚠️ **Disclaimer:** This is an automated preliminary assessment for educational demo purposes. "
                "It does not substitute for professional medical advice, diagnosis, or treatment."
            )
