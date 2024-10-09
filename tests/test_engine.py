import unittest

from helpers.app_config import Configuration
from helpers.embedding_factory import EmbeddingFactory
from helpers.engine_factory import EngineFactory
from helpers.vector_factory import VectorFactory
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


class TestEngine(unittest.TestCase):
    def test_get_engine_with_huggingface_baai_embedding(self):
        config = Configuration()

        embedding_factory = EmbeddingFactory("embedding-model")

        # Not using values from configuration because to intentionally test the particular embedding-model.
        embedding_model = embedding_factory.get_huggingface_embedding("BAAI/bge-base-en-v1.5")
        self.assertEqual(embedding_model.model_name,"BAAI/bge-base-en-v1.5")

        vector_factory = VectorFactory(config.config_values["llm_name"],
                                embedding_model,
                                config.config_values["llm_temperature"],
                                config.config_values["app_progress"])

        # Returns an updated global setting configuration that need to be applied when required.R
        index, settings = vector_factory.get_vector_index(
            config.config_values["app_data"],
            config.config_values["app_metadata"],
            config.config_values["chunk_size"],
            config.config_values["chunk_overlap"])
        self.assertIsNotNone(index.vector_store)
        self.assertEqual(settings.embed_model, embedding_model)

        #  Manually creating an engine to assert.
        engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=config.config_values["app_verbose"]
        )

        response = engine.query("who is appa?")
        self.assertIsNotNone(response)

        engine_factory = EngineFactory()

        retriever_engine = engine_factory.get_query_retriever(index,
                                                              config.config_values["ret_max"],
                                                              config.config_values["ret_score"],
                                                              config.config_values["app_verbose"])

        # Assert response is not null.
        response = retriever_engine.query("Who is Appa?")
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
        config = Configuration()

        embedding_factory = EmbeddingFactory("embedding-model")

        # Not using values from configuration because to intentionally test the particular embedding-model.
        embedding_model = embedding_factory.get_ollama_embedding("llama3.1")
        self.assertEqual(embedding_model.model_name,"llama3.1")

        # factory = VectorFactory("llama3.1",embedding_model,True)
        vector_factory = VectorFactory(config.config_values["llm_name"],
                                embedding_model,
                                config.config_values["llm_temperature"],
                                config.config_values["app_progress"])

        # Returns an updated global setting configuration that need to be applied when required.
        index, settings = vector_factory.get_vector_index(
            config.config_values["app_data"],
            config.config_values["app_metadata"],
            config.config_values["chunk_size"],
            config.config_values["chunk_overlap"])
        self.assertIsNotNone(index.vector_store)
        self.assertEqual(settings.embed_model, embedding_model)

        #  Manually creating an engine to assert.
        engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=config.config_values["app_verbose"]
        )

        response = engine.query("who is appa?")
        self.assertIsNotNone(response)

        engine_factory = EngineFactory()

        retriever_engine = engine_factory.get_query_retriever(index,
                                                              config.config_values["ret_max"],
                                                              config.config_values["ret_score"],
                                                              config.config_values["app_verbose"])

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


if __name__ == '__main__':
    unittest.main()
