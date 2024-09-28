from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


def get_vector_store(model_name: str, data_dir:str, chunk_size: int = None):
    documents = SimpleDirectoryReader(data_dir).load_data()

    # This code was disabled because this only works for OpenAI.
    # index=VectorStoreIndex.from_documents(documents,show_progress=True)
    # Remember! Don't use ServiceContext it was depreciated and replaced with Settings.
    Settings.embed_model = HuggingFaceEmbedding(
        cache_folder="embedding-model",
        model_name=model_name,
    )

    if chunk_size is not None:
        Settings.chunk_size = chunk_size

    # The llm should run because the API endpoint will be called for embeddings.
    Settings.llm = Ollama(model="llama3.1", request_timeout=360.0)

    # Vector Store Index turns all of your text into embeddings using an API from your LLM
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    # Returns a query engine.
    # return index.as_query_engine(
    #     text_qa_template=text_qa_template,
    #     refine_template=refine_template
    # )

    # return index.as_chat_engine(
    #     chat_mode="condense_question",
    #     verbose=True
    # )
    return index

    # Returns a query engine.
    # retriever = VectorIndexRetriever(index=index,similarity_top_k=4)
    # return RetrieverQueryEngine(retriever=retriever)

if __name__ == '__main__':
    question = """
    What makes Appa to pursue a career in teaching? 
    # """
    try:
        vector_store = get_vector_store("BAAI/bge-base-en-v1.5", "data")
        chat_engine = vector_store.as_chat_engine(
                chat_mode="condense_question",
                verbose=True
        )
        response = chat_engine.chat(question)
        print(response)
        # query_engine2 = get_query_engine("sentence-transformers/all-mpnet-base-v2", 1024)
    except Exception as e:
        print("Exception: ---->", e)