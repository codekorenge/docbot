import unittest
from unittest import TestCase

from helpers.llama_helper import *

class Test(unittest.TestCase):
    def test_get_vector_store(self):
        index, settings = get_vector_index("BAAI/bge-base-en-v1.5", "../data", 200)
        engine = get_chat_engine(index, settings, 3900)
        response = engine.chat("Who is Appa?")
        pprint(response)

    def test_get_query_engine(self):
        index, settings = get_vector_index("BAAI/bge-base-en-v1.5", "../data", 200)
        engine = get_query_engine(index)
        response = engine.query("Who is Appa?")
        pprint(response)

    def test_get_query_retriever(self):
        # index, settings = get_vector_index("BAAI/bge-base-en-v1.5",
        #                                    "../data",
        #                                    500,
        #                                    50)

        # engine= index.as_query_engine(settings = settings)
        engine = get_query_retriever(10, 0.10)
        response = engine.query("Who is Appa?")
        for node in response.source_nodes:
            print(f"score:{node.score}, "
                  f"word-count:{len(node.text.split())}, "
                  # f"node-meta:{node.metadata}, "
                  f"\tdocument:{node.metadata['file_name']}, "
                  f"\ttext:{node.text}")

if __name__ == '__main__':
    unittest.main()

