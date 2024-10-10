import streamlit as st
from datetime import datetime

import app
from helpers.metrics import display_retrieval_metrics

st.title(app.config.config_values["app_name"] +"- chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display only once.
user = st.session_state["user"]
if not app.greeted:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    st.markdown(f"### Hello {user}, it's {current_time} now.")
    app.greeted = True

# Accept user input
prompt = st.chat_input("ask me something")
if prompt:
    print(prompt)
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    print("-------------------[RETRIEVAL-START]--------------------\r\n")

    response = app.chatbot_engine.stream_chat(prompt)

    with st.chat_message("assistant"):
        if len(response.source_nodes) == 0:
            st.write(f"Sorry {user}, can't find any information regarding that in the local corpus.")
        else:
            st.write_stream(response.response_gen)

    display_retrieval_metrics(response, app.config)
    print("--------------------[RETRIEVAL-END]---------------------\r\n")

    st.session_state.messages.append({"role": "assistant", "content": response})