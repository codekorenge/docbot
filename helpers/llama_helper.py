from pprint import pprint
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate

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
    ), Settings

    return index

def get_chat_engine(vector_store: VectorStoreIndex, settings: Settings, token_limit: int):
    # memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)

    return vector_store.as_chat_engine(
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
def get_query_engine(vector_store: VectorStoreIndex):
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
    return vector_store.as_query_engine(
        response_mode="tree_summarize",
        verbose=True
    )