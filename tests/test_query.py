import unittest

from helpers.app_config import Configuration
from helpers.embedding_factory import EmbeddingFactory
from helpers.engine_factory import EngineFactory
from helpers.vector_factory import VectorFactory
from helpers.metrics import display_retrieval_metrics

class TestQuery(unittest.TestCase):


    def test_get_chat_engine_with_huggingface_baai_embedding(self):
        config = Configuration()

        embedding_factory = EmbeddingFactory("embedding-model")

        embedding_model = embedding_factory.get_huggingface_embedding(config.config_values["embed_name"])
        self.assertEqual(embedding_model.model_name, config.config_values["embed_name"])

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

        engine_factory = EngineFactory()

        retriever_engine = engine_factory.get_query_retriever(index,
                                                              config.config_values["ret_max"],
                                                              config.config_values["ret_score"],
                                                              config.config_values["app_verbose"])
        chat_engine = engine_factory.get_context_chat_engine(retriever_engine,
                                                             config.config_values["llm_token_limit"],
                                                             config.config_values["app_prompts"],
                                                             config.config_values["app_verbose"])

        print("-------------------[RETRIEVAL-START]--------------------\r\n")

        response = chat_engine.chat("who is appa and how old he is now?")
        display_retrieval_metrics(response, config)
        print(f"Answer: {response}")

        print("--------------------[RETRIEVAL-END]---------------------\r\n")
        # chat_engine.chat_repl()

if __name__ == '__main__':
    unittest.main()