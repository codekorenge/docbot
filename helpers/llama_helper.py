from pprint import pprint
from tabnanny import verbose
from typing import Callable

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate
from llama_index.core.node_parser import SentenceSplitter

"""
This code utilizes api/embeddings endpoint to generate running model's embedding.
This code was taken from the following URL.
https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/
"""
# def get_ollama_embedding(model_name: str) -> OllamaEmbedding:
#     return OllamaEmbedding(
#         model_name=model_name,
#         base_url="http://localhost:11434",
#         ollama_additional_kwargs={"mirostat": 0},
#     )
#
# def get_text_embedding(embedding: OllamaEmbedding, text:str)-> list[float] :
#     return embedding.get_text_embedding(text)

# def get_vector_index1(model_name: str,
#                      embedding_model,
#                      data_dir:str,
#                      chunk_size: int = None,
#                      chunk_overlap: int = None ):
#
#     # text = open(corpus_path).read()
#     # documents = Document(
#     #     text=text,
#     #     metadata={
#     #         "file_name": "01",
#     #         "category": "story",
#     #         "author": "star",
#     #     },
#     #     excluded_llm_metadata_keys=["file_name"],
#     #     metadata_seperator="::",
#     #     metadata_template="{key}=>{value}",
#     #     text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
#     # )
#     documents = SimpleDirectoryReader(data_dir).load_data()
#
#     # This code was disabled because this only works for OpenAI.
#     # index=VectorStoreIndex.from_documents(documents,show_progress=True)
#     # Remember! Don't use ServiceContext it was depreciated and replaced with Settings.
#     # Settings.embed_model = embedding_model
#     Settings.embed_model = HuggingFaceEmbedding(
#         cache_folder="embedding-model",
#         model_name=model_name,
#         # embed_batch_size=3072 (for llama3.1)
#     )
#
#     if chunk_size is not None:
#         Settings.chunk_size = chunk_size
#
#     if chunk_overlap is not None:
#         Settings.chunk_overlap = chunk_overlap
#
#     # The llm should run because the API endpoint will be called for embeddings.
#     Settings.llm = Ollama(model=embedding_model,
#                           temperature=0.1,
#                           request_timeout=360.0)
#
#     # Vector Store Index turns all of your text into embeddings using an API from your LLM
#     index = VectorStoreIndex.from_documents(
#         documents,
#         show_progress=True,
#     )
#
#     return index, Settings

def get_vector_index(model_name: str, data_dir:str, chunk_size: int = None, chunk_overlap: int = None ):
    documents = SimpleDirectoryReader(data_dir).load_data()

    # This code was disabled because this only works for OpenAI.
    # index=VectorStoreIndex.from_documents(documents,show_progress=True)
    # Remember! Don't use ServiceContext it was depreciated and replaced with Settings.
    Settings.embed_model = HuggingFaceEmbedding(
        cache_folder="embedding-model",
        model_name=model_name,
        # embed_batch_size=3072 (for llama3.1)
    )

    if chunk_size is not None:
        Settings.chunk_size = chunk_size

    if chunk_overlap is not None:
        Settings.chunk_overlap = chunk_overlap

    # The llm should run because the API endpoint will be called for embeddings.
    Settings.llm = Ollama(model="llama3.1",
                          temperature=0.1,
                          request_timeout=360.0)

    # Vector Store Index turns all of your text into embeddings using an API from your LLM
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    return index, Settings

def get_query_retriever(similarity_top_k:int,
                        similarity_cutoff: float ) -> RetrieverQueryEngine:
    # configure retriever
    documents = SimpleDirectoryReader("../data").load_data()

    # This code was disabled because this only works for OpenAI.
    # index=VectorStoreIndex.from_documents(documents,show_progress=True)
    # Remember! Don't use ServiceContext it was depreciated and replaced with Settings.
    Settings.embed_model = HuggingFaceEmbedding(
        cache_folder="embedding-model",
        model_name="BAAI/bge-base-en-v1.5",
        # embed_batch_size=3072 (for llama3.1)
    )

    # if chunk_size is not None:
    #     Settings.chunk_size = chunk_size
    #
    # if chunk_overlap is not None:
    #     Settings.chunk_overlap = chunk_overlap

    # The llm should run because the API endpoint will be called for embeddings.
    Settings.llm = Ollama(model="llama3.1",
                          temperature=0.1,
                          request_timeout=360.0)

    # Vector Store Index turns all of your text into embeddings using an API from your LLM
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )


    # configure response synthesizer
    response_synthesizer = get_response_synthesizer()

    # assemble query engine
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
    )

def get_chat_engine(index: VectorStoreIndex, settings: Settings, token_limit: int):
    # memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)

    return index.as_chat_engine(
        chat_mode='condense_plus_context',
        llm=settings.llm,
        memory=memory,
        context_prompt=(
            "You are a chatbot, able to have normal interactions, as well as talk"
            "about documents in the database. Always explain queries from a third person perspective."
            "Here are the relevant documents for the context:\n"
            "{context_str}"
            "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."),
        verbose=True
    )

# TODO: Need revision!
def get_query_engine(index: VectorStoreIndex):
    # qa_prompt_str = (
    #     "Context information is below.\n"
    #     "---------------------\n"
    #     "{context_str}\n"
    #     "---------------------\n"
    #     "Given the context information and not prior knowledge, "
    #     "answer the question: {query_str}\n"
    # )
    #
    # refine_prompt_str = (
    #     "We have the opportunity to refine the original answer "
    #     "(only if needed) with some more context below.\n"
    #     "------------\n"
    #     "{context_msg}\n"
    #     "------------\n"
    #     "Given the new context, refine the original answer to better "
    #     "answer the question: {query_str}. "
    #     "If the context isn't useful, output the original answer again.\n"
    #     "Original Answer: {existing_answer}"
    # )
    #
    # # Text QA Prompt
    # chat_text_qa_msgs = [
    #     ChatMessage(
    #         role=MessageRole.SYSTEM,
    #         content=(
    #             "Always answer the question, even if the context isn't helpful."
    #         ),
    #     ),
    #     ChatMessage(role=MessageRole.USER, content=qa_prompt_str),
    # ]
    # text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
    #
    # # Refine Prompt
    # chat_refine_msgs = [
    #     ChatMessage(
    #         role=MessageRole.SYSTEM,
    #         content=(
    #             "Always answer the question, even if the context isn't helpful."
    #         ),
    #     ),
    #     ChatMessage(role=MessageRole.USER, content=refine_prompt_str),
    # ]
    # refine_template = ChatPromptTemplate(chat_refine_msgs)

    # Returns a query engine.
    return index.as_query_engine(
        response_mode="tree_summarize",
        verbose=True
    )