from sqlalchemy import false

from helpers.app_config import Configuration
import streamlit as st
from llama_index.core.chat_engine import CondenseQuestionChatEngine

from helpers.app_config import Configuration

# create application configuration.
config =Configuration()

# Only initialized after login.
chatbot_engine : CondenseQuestionChatEngine

initialized = False

greeted = False

if __name__ == '__main__':
    st.switch_page('pages/login.py')


