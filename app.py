from helpers.llama_helper import get_vector_store, get_chat_engine
import streamlit as st


if __name__ == '__main__':
    # Get LlamaIndex vector-store and chat-engine.
    vector_store, settings= get_vector_store("BAAI/bge-base-en-v1.5", "data", 200)
    chat_engine = get_chat_engine(vector_store, settings, 3900)

    st.title("{ docbot }")

    # Set a default model
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "llama3.5"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    prompt = st.chat_input("ask me something")
    if prompt:
        print(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = chat_engine.stream_chat(prompt)

        with st.chat_message("assistant"):
            st.write_stream(response.response_gen)

        st.session_state.messages.append({"role": "assistant", "content": response})