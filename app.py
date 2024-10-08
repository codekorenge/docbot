from helpers.embedding_factory import EmbeddingFactory
from helpers.engine_factory import EngineFactory
from helpers.llama_helper import get_vector_index, get_chat_engine
import streamlit as st

from helpers.vector_factory import VectorFactory

if __name__ == '__main__':
    # Get LlamaIndex vector-store and chat-engine.
    # vector_store, settings= get_vector_index("BAAI/bge-base-en-v1.5", "data", 200)
    # chat_engine = get_chat_engine(vector_store, settings, 3900)

    # Creating the embedding model for transforming corpus into vector embeddings.
    embedding_factory = EmbeddingFactory("embedding-model")
    embedding_model = embedding_factory.get_huggingface_embedding("BAAI/bge-base-en-v1.5")

    # The embedding model will be used to read the corpus inside the directory and transforming into vector embeddings.
    factory = VectorFactory("llama3.1", embedding_model, True)
    index, settings = factory.get_vector_index("data", 200, 10)

    # The retriever is configured to retrieve K chunks.
    engine_factory = EngineFactory()
    retriever_engine = engine_factory.get_query_retriever(index, 10, 0.55)
    chat_engine = engine_factory.get_context_chat_engine(retriever_engine, 1500)

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
            if len(response.source_nodes) ==0:
                st.write("Sorry, can't find any information regarding that in the local corpus.")
            else:
                st.write_stream(response.response_gen)

        st.session_state.messages.append({"role": "assistant", "content": response})