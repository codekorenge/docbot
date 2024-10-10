import streamlit as st

import app
from helpers import chatbot_factory
from helpers.app_config import Configuration
from helpers.embedding_factory import EmbeddingFactory
from helpers.engine_factory import EngineFactory
from helpers.llama_helper import get_vector_index, get_chat_engine
from llama_index.core.chat_engine import CondenseQuestionChatEngine, CondensePlusContextChatEngine

gh : int


if not app.initialized:
    st.session_state["user"] = ""

if st.button('Login'):
    st.session_state["user"] = "codek"
    # Create chatbot engine from the configuration.
    app.chatbot_engine = chatbot_factory.create_chatbot(app.config)
    app.initialized = True
    st.switch_page('pages/chatbot.py')
