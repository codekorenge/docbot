from unittest import TestCase

from helpers.llama_helper import *


class Test(TestCase):
    def test_get_vector_store(self):
        vector_store, settings = get_vector_store("BAAI/bge-base-en-v1.5", "../data", 200)
        engine = get_chat_engine(vector_store, settings, 3900)
        response = engine.chat("Who is Appa?")
        pprint(response)

    def test_get_query_engine(self):
        vector_store, settings = get_vector_store("BAAI/bge-base-en-v1.5", "../data", 200)
        engine = get_query_engine(vector_store)
        response = engine.query("Who is Appa?")
        pprint(response)

