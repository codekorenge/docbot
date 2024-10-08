from lib2to3.fixes.fix_input import context

from llama_index.core import PromptTemplate
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.chat_engine import CondenseQuestionChatEngine, CondensePlusContextChatEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer

class EngineFactory:
    def __init__(self):
        pass
    """
    Code was referenced from here: https://docs.llamaindex.ai/en/stable/understanding/querying/querying/
    """
    def get_query_retriever(self,
                            index: VectorStoreIndex,
                            similarity_top_k: int,
                            similarity_cutoff: float,
                            verbose: bool) -> RetrieverQueryEngine:

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            verbose=verbose
        )

        # configure response synthesizer
        # response_synthesizer = get_response_synthesizer(verbose=True)
        response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            verbose=verbose
        )

        # assemble query engine
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )

    """
    Code was referenced from: https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern/
    """
    def get_context_chat_engine(self,
                                retriever: RetrieverQueryEngine,
                                token_limit:int,
                                prompts:str,
                                verbose: bool) -> CondensePlusContextChatEngine:
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

        system_prompt = open( prompts + "system_prompt.txt", "r").read()
        context_prompt = open(prompts + "context_prompt.txt", "r").read()

        # Swapped the system and context places because context is inserted first.
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            system_prompt=context_prompt,
            context_prompt=system_prompt,
            memory=memory,
            # condense_prompt=custom_prompt,
            # condense_question_prompt=custom_prompt,
            # chat_history=custom_chat_history,
            verbose=verbose,
        )

        return chat_engine