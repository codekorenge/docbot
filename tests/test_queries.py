import unittest
from typing import overload

from helpers.llama_helper import *


class TestQueries(unittest.TestCase):
    def setUp(self):
        self.name = "test"
        # self.embedding_model = get_ollama_embedding("llama3.1")
        # self.index = get_vector_index("BAAI/bge-base-en-v1.5", "../data", 200, 50)

    def test_question1(self):
        # index = get_vector_index("BAAI/bge-base-en-v1.5", "../data", 200, 50)
        retriever_engine = get_query_retriever(similarity_top_k=10,
                                               similarity_cutoff=.20)

        response = retriever_engine.query("Who is appa?")
        print(response)


if __name__ == '__main__':
    unittest.main()
