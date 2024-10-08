from lib2to3.fixes.fix_input import context

from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.chat_engine import CondenseQuestionChatEngine, CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer

class EngineFactory:
    def __init__(self):
        pass
    """
    Code was referenced from here: https://docs.llamaindex.ai/en/stable/understanding/querying/querying/
    """
    def get_query_retriever(self,
                            index: VectorStoreIndex,
                            similarity_top_k: int,
                            similarity_cutoff: float) -> RetrieverQueryEngine:

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
        )

        # configure response synthesizer
        response_synthesizer = get_response_synthesizer()

        # assemble query engine
        return RetrieverQueryEngine(
            retriever=retriever,
            # response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )

    """
    Code was referenced from: https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern/
    """
    def get_context_chat_engine(self, retriever: RetrieverQueryEngine, token_limit:int) -> CondensePlusContextChatEngine:
        memory = ChatMemoryBuffer.from_defaults(token_limit=token_limit)

        # custom_prompt = PromptTemplate(
        #     """\
        # Given a conversation (between Human and Assistant) and a follow up message from Human, \
        # rewrite the message to be a standalone question that captures all relevant context \
        # from the conversation.
        #
        # <Chat History>
        # {chat_history}
        #
        # <Follow Up Message>
        # {question}
        #
        # <Standalone question>
        # """
        # )

        # list of `ChatMessage` objects
        # custom_chat_history = [
        #     ChatMessage(
        #         role=MessageRole.USER,
        #         content="Hello assistant, we are having a insightful discussion about Paul Graham today.",
        #     ),
        #     ChatMessage(role=MessageRole.ASSISTANT, content="Okay, sounds good."),
        # ]

        # chat_engine = CondenseQuestionChatEngine.from_defaults(
        #     query_engine=retriever,
        #     condense_question_prompt=custom_prompt,
        #     chat_history=custom_chat_history,
        #     verbose=True,
        # )

        system_prompt=("You are a chatbot, able to have normal interactions, as well as talk about documents "
                       "in the database. Always explain queries from a third person perspective.")
        context_prompt= ("Here are the relevant documents for the context:\n"
                         "{context_str}"
                         "\nInstruction: Use the previous chat history, or the context above, to interact and help the user.")

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            system_prompt=system_prompt,
            context_prompt=context_prompt,
            memory=memory,
            # condense_prompt=custom_prompt,
            # condense_question_prompt=custom_prompt,
            # chat_history=custom_chat_history,
            verbose=True,
        )

        return chat_engine