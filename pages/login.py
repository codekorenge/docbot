import time

import streamlit as st

import app
from helpers import chatbot_factory

st.title("docbot - login")

# Create an empty container
placeholder = st.empty()

# A dummy list for authenticating sets of users.
credentials = {"codek": "123", "simpson": "123"}

# Insert a form in the container
with placeholder.form("login"):
    st.markdown("### Please enter your credentials")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit and username in credentials.keys():
    if credentials[username] == password:
        st.session_state.user = username
        st.session_state.password = password
        placeholder.empty()
        st.success("Login successful ...")

        time.sleep(3)
        st.success("Initializing chatbot ...")
        time.sleep(1)
        data_dir = app.config.config_values["app_data"]
        st.success(f"loading corpus from [{data_dir}] directory ...")
        app.chatbot_engine = chatbot_factory.create_chatbot(app.config)
        app.initialized = True
        st.switch_page('pages/chatbot.py')
    else:
        st.error("Login failed")
else:
    pass