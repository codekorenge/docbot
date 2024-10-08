from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


class EngineFactory:
    def __init__(self):
        pass

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
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)]
        )