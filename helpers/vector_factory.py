from importlib.metadata import metadata
from statistics import median
from xml.dom.minidom import Document
import json
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core import VectorStoreIndex, schema
from llama_index.llms.ollama import Ollama
import nltk

class VectorFactory:
    # TODO: embedding_model param has not strong name typing!
    def __init__(self,
                 model_name: str,
                 embedding_model,
                 temperature: float,
                 show_progress: bool):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.cache_folder = "embedding-model"
        self.temperature = temperature
        self.request_timeout = 360.0
        self.show_progress = show_progress

    def __load_metadata(self,
                        filename:str):
        # Open and read the JSON file
        with open(filename, 'r') as file:
            data = json.load(file)
        file.close()
        return data

    def __load_documents(self,
                         data_dir: str,
                         metadata_dir: str):
        print("<<<<<<<<<< Loading Corpus >>>>>>>>>>")

        # Load documents from directory.
        documents = (SimpleDirectoryReader(data_dir)
                     .load_data())

        print("Checking metadata-switch...")

        corpus_text = ""
        if metadata_dir is None:
            cnt = 0
            for document in documents:
                cnt += 1
                print(f"doc-name: [{cnt}], \ttitle: {document.metadata['file_name']}");
                corpus_text += document.get_content()
        else:
            print("Metadata-switch found, loading metadata...")
            # Load metadata for each documents.
            for document in documents:
                meta = self.__load_metadata(metadata_dir + "/" + document.metadata['file_name'] + ".json")
                document.metadata["title"] = meta["title"]
                document.metadata["description"] = meta["description"]

            cnt = 0
            for document in documents:
                cnt += 1
                print(f"doc-name: [{cnt}], \ttitle: {document.metadata['title']}")
                print(f"\tdescription: {document.metadata['description']}");
                corpus_text += document.get_content()

        return documents, corpus_text

    def __display_chunk_info(self,
                             index: VectorStoreIndex,
                             chunk_size: int,
                             chunk_overlap: int):
        # Display metric-2: Chunks
        # Info not documented in API. Experimental discovery. In the dict returned by ref_doc_info,
        # each info created for a doc (docs are for every document). Inside the doc, item with index 1
        # holds the metadata and node_ids.
        docs = index.ref_doc_info.items()
        doc_list = {}
        for doc in docs:
            node_info = doc[1]
            doc_name = node_info.metadata["file_name"]
            doc_list[doc_name] = node_info.node_ids

        total_word_cnt = 0
        total_chunk_cnt = 0
        total_word_cnt_list = []
        for doc_name in doc_list:
            node_ids = doc_list[doc_name]
            node_counter = 0
            node_word_cnt = 0
            node_word_cnt_list = ""
            for node_id in node_ids:
                node_counter += 1
                node = index.docstore.get_node(node_id)
                node_word_cnt += len(node.text.split())
                node_word_cnt_list += str(len(node.text.split())) + ", "
                total_word_cnt_list.append(str(len(node.text.split())))

                # Interim: use words not token for calculation.
                # tokens = nltk.word_tokenize(node.text)
                # node_word_cnt_list += str(len(tokens)) + ", "

            total_word_cnt += node_word_cnt
            total_chunk_cnt += node_counter
            node_word_cnt_list = node_word_cnt_list[:-2]

            print(f"doc-name: [{doc_name}], total-chunk: [{node_counter}] and total-word: [{node_word_cnt}].")
            print(f"\tList of word-count for each chunk:[{node_word_cnt_list}].")

        print(f"\r\nFor entire corpus, total-node: [{total_chunk_cnt}], and total-word: [{total_word_cnt}].")
        print(f"Configured params, chunk-size: [{chunk_size}], and chunk-overlap: [{chunk_overlap}].")
        print(f"\r\nFormula                 \t\t: total-word / chunk-size ~= chunk-count.")
        print(f"Expected                \t\t: {total_word_cnt} / {chunk_size} = {total_word_cnt / chunk_size}.")
        # print(f"Actual                          \t: {total_word_cnt} / {chunk_size} ~= {total_chunk_cnt}.")
        calculated_chuck_ave = total_word_cnt / len(total_word_cnt_list)
        print(f"Actual (ave(chunk-size)) \t\t: {total_word_cnt} / {calculated_chuck_ave} ~= {total_word_cnt/calculated_chuck_ave}.")

    def __display_chunk_info_in_token(self,
                             index: VectorStoreIndex,
                             chunk_size: int,
                             chunk_overlap: int):
        # Display metric-2: Chunks
        # Info not documented in API. Experimental discovery. In the dict returned by ref_doc_info,
        # each info created for a doc (docs are for every document). Inside the doc, item with index 1
        # holds the metadata and node_ids.
        docs = index.ref_doc_info.items()
        doc_list = {}
        for doc in docs:
            node_info = doc[1]
            doc_name = node_info.metadata["file_name"]
            doc_list[doc_name] = node_info.node_ids

        total_word_cnt = 0
        total_chunk_cnt = 0
        total_word_cnt_list = []
        for doc_name in doc_list:
            node_ids = doc_list[doc_name]
            node_counter = 0
            node_word_cnt = 0
            node_word_cnt_list = ""
            for node_id in node_ids:
                node_counter += 1
                node = index.docstore.get_node(node_id)

                tokens = nltk.word_tokenize(node.text)
                node_word_cnt += len(tokens)
                # print(len(node.text.split()))
                # print(len(tokens))

                node_word_cnt_list += str(len(tokens)) + ", "
                total_word_cnt_list.append(str(len(tokens)))

                # Interim: use words not token for calculation.
                # tokens = nltk.word_tokenize(node.text)
                # node_word_cnt_list += str(len(tokens)) + ", "

            total_word_cnt += node_word_cnt
            total_chunk_cnt += node_counter
            node_word_cnt_list = node_word_cnt_list[:-2]

            print(f"Doc[{doc_name}], total-chunk: [{node_counter}] and total-word: [{node_word_cnt}].")
            print(f"\tList of word-count for each chunk:[{node_word_cnt_list}].")

        print(f"\r\nFor entire corpus: total-node: [{total_chunk_cnt}] and total-word: [{total_word_cnt}].")
        print(f"Configured params: chunk-size: [{chunk_size}] and chunk-overlap: [{chunk_overlap}].")
        print(f"\r\nFormula                     \t: total-word / chunk-size ~= chunk-count.")
        print(f"Expected                        \t: {total_word_cnt} / {chunk_size} = {total_word_cnt / chunk_size}.")
        # print(f"Actual                          \t: {total_word_cnt} / {chunk_size} ~= {total_chunk_cnt}.")
        calculated_chuck_ave = total_word_cnt / len(total_word_cnt_list)
        print(
            f"Actual (ave(chunk-size))        \t: {total_word_cnt} / {calculated_chuck_ave} ~= {total_word_cnt / calculated_chuck_ave}.")

    def get_vector_index(self,
                         data_dir: str,
                         metadata_dir: str,
                         chunk_size: int,
                         chunk_overlap: int):
        print("--------------------[INDEXING-START]--------------------\r\n")

        # load corpus from directories.
        documents, corpus_text = self.__load_documents(data_dir, metadata_dir)

        # Display metric-1: Document
        print(f"Total-doc  \t:{len(documents)}")
        print(f"Total-word \t:{len(corpus_text.split())}\r\n")

        print("<<<<<<<<< Creating chunks >>>>>>>>>>")
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

        self.__display_chunk_info(index, chunk_size, chunk_overlap)

        print("---------------------[INDEXING-END]---------------------\r\n")

        return index, Settings