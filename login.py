import streamlit as st

st.set_page_config(page_title="Login", page_icon="🏥")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.title("🏥 AI Healthcare Chatbot Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):

    if username == "admin" and password == "admin123":
        st.session_state.logged_in = True
        st.success("Login Successful!")

    else:
        st.error("Invalid Username or Password")

if st.session_state.logged_in:
    st.info("Login successful. Open chatbot_app.py to use the application.")
