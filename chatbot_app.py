import os
import pickle
import streamlit as st

# Force absolute pathing to find the file safely on the cloud server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "disease_model.pkl")


@st.cache_resource
def load_model():
    try:
        # Using standard pickle but wrapped safely to catch errors
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# Simple UI Layout
st.title("🩺 Disease Predictor Chatbot")

if model is not None:
    st.success("Model loaded successfully!")
    # Your prediction inputs go here...
else:
    st.error("Model failed to load. Please check your Python version compatibility.")
