import unittest
from symbol import factor

from helpers.embedding_factory import EmbeddingFactory
from helpers.engine_factory import EngineFactory
from helpers.vector_factory import VectorFactory

class TestEngineFactory(unittest.TestCase):
    def test_get_engine_with_huggingface_baai_embedding(self):
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding_model = embedding_factory.get_huggingface_embedding("BAAI/bge-base-en-v1.5")
        self.assertEqual(embedding_model.model_name,"BAAI/bge-base-en-v1.5")

        factory = VectorFactory("llama3.1",embedding_model,True)

        # Returns an updated global setting configuration that need to be applied when required.
        index, settings = factory.get_vector_index("../data",200,10)
        self.assertIsNotNone(index.vector_store)
        self.assertEqual(settings.embed_model, embedding_model)

        #  Manually creating an engine to assert.
        engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True
        )

        response = engine.query("who is appa?")
        self.assertIsNotNone(response)

        engine_factory = EngineFactory()

        engine = engine_factory.get_query_retriever(index,10, 0.50)

        # Assert response is not null.
        response = engine.query("Who is Appa?")
        self.assertIsNotNone(response)
        print(f"response: {response}")

        self.assertGreater(len(response.source_nodes),0)

        cnt=0
        for node in response.source_nodes:
            cnt +=1
            print(f"{cnt}."
                  f"\tscore:{node.score}, "
                  f"\tword-count:{len(node.text.split())}, "
                  # f"node-meta:{node.metadata}, "
                  f"\tdocument:{node.metadata['file_name']}, "
                  f"\ttext:{node.text[0:30]} ...")


    def test_get_engine_with_ollama_embedding(self):
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding_model = embedding_factory.get_ollama_embedding("llama3.1")
        self.assertEqual(embedding_model.model_name,"llama3.1")

        factory = VectorFactory("llama3.1",embedding_model,True)

        # Returns an updated global setting configuration that need to be applied when required.
        index, settings = factory.get_vector_index("../data",200,10)
        self.assertIsNotNone(index.vector_store)
        self.assertEqual(settings.embed_model, embedding_model)

        #  Manually creating an engine to assert.
        engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True
        )

        response = engine.query("who is appa?")
        self.assertIsNotNone(response)

        engine_factory = EngineFactory()

        engine = engine_factory.get_query_retriever(index,10, 0.40)

        # Assert response is not null.
        response = engine.query("Who is Appa?")
        self.assertIsNotNone(response)
        print(f"response: {response}")

        self.assertGreater(len(response.source_nodes),0)

        cnt=0
        for node in response.source_nodes:
            cnt +=1
            print(f"{cnt}."
                  f"\tscore:{node.score}, "
                  f"\tword-count:{len(node.text.split())}, "
                  # f"node-meta:{node.metadata}, "
                  f"\tdocument:{node.metadata['file_name']}, "
                  f"\ttext:{node.text[0:30]} ...")

    def test_get_chat_engine_with_huggingface_baai_embedding(self):
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding_model = embedding_factory.get_huggingface_embedding("BAAI/bge-base-en-v1.5")
        self.assertEqual(embedding_model.model_name, "BAAI/bge-base-en-v1.5")

        factory = VectorFactory("llama3.1", embedding_model, True)

        # Returns an updated global setting configuration that need to be applied when required.
        index, settings = factory.get_vector_index("../data", 200, 10)
        self.assertIsNotNone(index.vector_store)
        self.assertEqual(settings.embed_model, embedding_model)

        # #  Manually creating an engine to assert.
        # engine = index.as_query_engine(
        #     response_mode="tree_summarize",
        #     verbose=True
        # )
        #
        # response = engine.query("who is appa?")
        # self.assertIsNotNone(response)

        engine_factory = EngineFactory()

        retriever_engine = engine_factory.get_query_retriever(index, 10, 0.55)
        chat_engine = engine_factory.get_context_chat_engine(retriever_engine, 1500)

        response = chat_engine.chat("who is appa and how old he is now?")
        # chat_engine.chat_repl()

        self.assertIsNotNone(response)
        print(f"response: {response}")

        cnt = 0
        for node in response.source_nodes:
            cnt += 1
            print(f"{cnt}."
                  f"\tscore:{node.score}, "
                  f"\tword-count:{len(node.text.split())}, "
                  # f"node-meta:{node.metadata}, "
                  f"\tdocument:{node.metadata['file_name']}, "
                  f"\ttext:{node.text[0:30]} ...")

if __name__ == '__main__':
    unittest.main()