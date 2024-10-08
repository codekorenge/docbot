from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama


class VectorFactory:
    # TODO: embedding_model param has not strong name typing!
    def __init__(self,
                 model_name: str,
                 embedding_model,
                 show_progress: bool = True):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.cache_folder = "embedding-model"
        self.temperature = 0.1
        self.request_timeout = 360.0
        self.show_progress = show_progress

    def get_vector_index(self,
                         data_dir: str,
                         chunk_size: int = None,
                         chunk_overlap: int = None):

        documents = (SimpleDirectoryReader(data_dir)
                     .load_data())

        # This code was disabled because this only works for OpenAI.
        # index=VectorStoreIndex.from_documents(documents,show_progress=True)
        # Remember! Don't use ServiceContext it was depreciated and replaced with Settings.
        Settings.embed_model = self.embedding_model

        if chunk_size is not None:
            Settings.chunk_size = chunk_size

        if chunk_overlap is not None:
            Settings.chunk_overlap = chunk_overlap

        # The llm should run because the API endpoint will be called for embeddings.
        Settings.llm = Ollama(model=self.model_name,
                              temperature=self.temperature,
                              request_timeout=self.request_timeout)

        # Vector Store Index turns all of your text into embeddings using an API from your LLM
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=self.show_progress,
        )

        return index, Settings