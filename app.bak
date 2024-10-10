from helpers.app_config import Configuration
from helpers.embedding_factory import EmbeddingFactory
from helpers.engine_factory import EngineFactory
from helpers.llama_helper import get_vector_index, get_chat_engine
from llama_index.core.chat_engine import CondenseQuestionChatEngine, CondensePlusContextChatEngine
import streamlit as st

from helpers.vector_factory import VectorFactory
from helpers.metrics import display_retrieval_metrics

@st.experimental_fragment
def reload_chat(engine: CondensePlusContextChatEngine):
    st.title(config.config_values['app_name'])

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

        print("-------------------[RETRIEVAL-START]--------------------\r\n")

        response = engine.stream_chat(prompt)

        with st.chat_message("assistant"):
            if len(response.source_nodes) == 0:
                st.write("Sorry, can't find any information regarding that in the local corpus.")
            else:
                st.write_stream(response.response_gen)

        display_retrieval_metrics(response, config)
        print("--------------------[RETRIEVAL-END]---------------------\r\n")

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    # Load application configuration from config.ini file.
    config = Configuration()

    # Creating the embedding model for transforming corpus into vector embeddings.
    embedding_factory = EmbeddingFactory("embedding-model")
    embedding_model = embedding_factory.get_huggingface_embedding(config.config_values["embed_name"])

    # The embedding model will be used to read the corpus inside the directory and transforming into vector embeddings.
    vector_factory = VectorFactory(config.config_values["llm_name"],
                                   embedding_model,
                                   config.config_values["llm_temperature"],
                                   config.config_values["app_progress"])

    # Returns an updated global setting configuration that need to be applied when required.
    index, settings = vector_factory.get_vector_index(config.config_values["app_data"],
                                                      config.config_values["app_metadata"],
                                                      config.config_values["chunk_size"],
                                                      config.config_values["chunk_overlap"])

    # The retriever is configured to retrieve K chunks.
    engine_factory = EngineFactory()
    retriever_engine = engine_factory.get_query_retriever(index,
                                                          config.config_values["ret_max"],
                                                          config.config_values["ret_score"],
                                                          config.config_values["app_verbose"])
    chat_engine = engine_factory.get_context_chat_engine(retriever_engine,
                                                         config.config_values["llm_token_limit"],
                                                         config.config_values["app_prompts"],
                                                         config.config_values["app_verbose"])

    reload_chat(chat_engine)

