from http.client import responses
from pprint import pprint

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import streamlit as st

def get_vector_store(model_name: str, data_dir:str, chunk_size: int = None):
    documents = SimpleDirectoryReader(data_dir).load_data()

    # This code was disabled because this only works for OpenAI.
    # index=VectorStoreIndex.from_documents(documents,show_progress=True)
    # Remember! Don't use ServiceContext it was depreciated and replaced with Settings.
    Settings.embed_model = HuggingFaceEmbedding(
        cache_folder="embedding-model",
        model_name=model_name,
    )

    if chunk_size is not None:
        Settings.chunk_size = chunk_size

    # The llm should run because the API endpoint will be called for embeddings.
    Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

    # Vector Store Index turns all of your text into embeddings using an API from your LLM
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    return index


if __name__ == '__main__':
    # Llama code.
    vector_store = get_vector_store("BAAI/bge-base-en-v1.5", "data")
    chat_engine = vector_store.as_chat_engine(
            chat_mode="condense_question",
            verbose=True
    )

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
            pprint(response)
            # print(response.sources)

        st.session_state.messages.append({"role": "assistant", "content": response})