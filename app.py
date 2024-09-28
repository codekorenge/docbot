from http.client import responses

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import ChatPromptTemplate


# %%
def get_query_engine(model_name: str, chunk_size: int = None):
    qa_prompt_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the question: {query_str}\n"
    )

    refine_prompt_str = (
        "We have the opportunity to refine the original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
    )

    # Text QA Prompt
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Always answer the question, even if the context isn't helpful."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=qa_prompt_str),
    ]
    text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

    # Refine Prompt
    chat_refine_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Always answer the question, even if the context isn't helpful."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=refine_prompt_str),
    ]
    refine_template = ChatPromptTemplate(chat_refine_msgs)

    documents = SimpleDirectoryReader("data").load_data()

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
        # show_progress=True,
    )

    # Returns a query engine.
    return index.as_query_engine(
        text_qa_template=text_qa_template,
        refine_template=refine_template
    )

    # Returns a query engine.
    # retriever = VectorIndexRetriever(index=index,similarity_top_k=4)
    # return RetrieverQueryEngine(retriever=retriever)

if __name__ == '__main__':
    question = """
    What makes Appa to pursue a career in teaching? 
    # """
    try:
        query_engine1 = get_query_engine("BAAI/bge-base-en-v1.5")
        # query_engine2 = get_query_engine("sentence-transformers/all-mpnet-base-v2", 1024)
    except Exception as e:
        print("Exception: ---->", e)

    response = query_engine1.query(question)
    print(response)