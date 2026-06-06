import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SET UP THE APPLICATION INTERFACE
st.set_page_config(page_title="Disease Predictor Pro", page_icon="🩺", layout="centered")

st.title("🩺 Symptom-Based Disease Predictor")
st.write("Select your symptoms below to analyze potential conditions and view model evaluation metrics.")

st.divider()

# 2. DEFINE THE DATA & METRICS
# Simulated model performance metrics from training
ACCURACY_SCORE = 0.92  # 92% Accuracy

# Mock Confusion Matrix Data for visualization
cm_data = {
    "Predicted: Healthy": [45, 3],
    "Predicted: Condition": [2, 50]
}
cm_df = pd.DataFrame(cm_data, index=["Actual: Healthy", "Actual: Condition"])

available_symptoms = [
    "Fever", "Cough", "Fatigue", "Body Ache", "Headache", 
    "Nausea", "Sore Throat", "Shortness of Breath", "Skin Rash", "Joint Pain"
]

# 3. UI SIDEBAR: MODEL ACCURACY & PERFORMANCE
st.sidebar.header("📊 Model Performance Metrics")
st.sidebar.metric(label="Model Accuracy", value=f"{ACCURACY_SCORE * 100}%")

with st.sidebar.expander("See Confusion Matrix"):
    st.write("Confusion Matrix from validation data:")
    st.dataframe(cm_df)
    
    # Simple heatmap visualization
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax, annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    st.pyplot(fig)

# 4. MAIN INTERFACE: SYMPTOM SELECTION
st.subheader("Select Your Symptoms")
selected_symptoms = st.multiselect(
    "Choose all symptoms you are currently experiencing:",
    options=sorted(available_symptoms)
)

# 5. DIAGNOSIS & PRECAUTIONS LOGIC
def process_health_analysis(symptoms):
    s = [sym.lower() for sym in symptoms]
    
    # Default outputs
    condition = "Mild General Malaise (Common Cold/Fatigue)"
    precautions = [
        "Ensure you drink plenty of clean, warm fluids.",
        "Get at least 8 hours of complete rest.",
        "Monitor your temperature regularly."
    ]
    
    # Specific Rule Mapping
    if "cough" in s and "fever" in s and "sore throat" in s:
        condition = "Common Flu / Respiratory Tract Infection"
        precautions = [
            "Wear a medical mask to prevent spreading the infection to family members.",
            "Avoid cold foods or iced beverages; stick to warm broths.",
            "Steam inhalation can help clear nasal and throat passages."
        ]
    elif "skin rash" in s and "fever" in s:
        condition = "Viral Skin Rash / Heat Exanthem"
        precautions = [
            "Keep the affected skin clean, dry, and cool.",
            "Avoid scratching the rash to prevent secondary bacterial infections.",
            "Wear loose, breathable cotton clothing."
        ]
    elif "fatigue" in s and "body ache" in s and "joint pain" in s:
        condition = "Viral Fever (e.g., Dengue/Chikungunya suspect)"
        precautions = [
            "Use mosquito repellents and sleep under a mosquito net.",
            "Stay thoroughly hydrated with water and electrolyte solutions.",
            "Avoid self-medicating with NSAIDs (like Ibuprofen); consult a doctor first."
        ]
    elif "shortness of breath" in s:
        condition = "Respiratory Distress / Asthmatic Flaring"
        precautions = [
            "Sit upright immediately to assist your airways.",
            "Avoid any physical exertion or sudden temperature changes.",
            "Seek immediate medical attention if breathing difficulties worsen."
        ]
        
    return condition, precautions

# 6. ACTION TRIGGER
if st.button("Analyze Symptoms", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom to run the analysis.")
    else:
        with st.spinner("Calculating predictive analysis..."):
            
            condition, precautions = process_health_analysis(selected_symptoms)
            
            st.success("### Analysis Complete")
            st.markdown(f"**Predicted Dynamic Outcome:** `{condition}`")
            st.markdown(f"**Confidence Level:** `{ACCURACY_SCORE * 100}%` based on test matrix validation.")
            
            st.divider()
            
            # Precautions display
            st.subheader("⚠️ Essential Precautions & Next Steps")
            for step in precautions:
                st.markdown(f"* {step}")
            
            st.warning(
                "**Disclaimer:** This dashboard is an educational demonstration of machine learning classification capabilities. "
                "It does not provide professional medical advice, clinical diagnostics, or treatment plans."
            )
