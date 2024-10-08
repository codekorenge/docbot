from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class EmbeddingFactory:
    def __init__(self, cache_folder: str):
        self.base_url = "http://localhost:11434"
        self.cache_folder = cache_folder
    """
    This code utilizes api/embeddings endpoint to generate running model's embedding.
    This code was taken from the following URL.
    https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/
    """
    def get_ollama_embedding(self, embedding_name: str) -> OllamaEmbedding:
        return OllamaEmbedding(
            model_name=embedding_name,
            base_url=self.base_url,
            ollama_additional_kwargs={"mirostat": 0},
        )

    def get_huggingface_embedding(self, embedding_name: str)-> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            cache_folder=self.cache_folder,
            model_name=embedding_name,
            # embed_batch_size=3072 (for llama3.1)
        )
