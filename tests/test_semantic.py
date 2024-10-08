import unittest

from helpers.embedding_factory import EmbeddingFactory


class TestSemantic(unittest.TestCase):
    def test_similarity_for_two_simple_words_with_ollama_embedding(self):
        text1 = "dog"
        text2 = "puppy"
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding = embedding_factory.get_ollama_embedding("llama3.1")

        a = embedding.get_text_embedding(text1)
        b= embedding.get_text_embedding(text2)

        score = embedding.similarity(a,b)
        print(f"similarity score is: {score}")

        self.assertGreater(score, 0.75)

    def test_similarity_for_two_simple_sentences_with_ollama_embedding(self):
        text1 = "dog is a canine."
        text2 = "a puppy is also a canine."
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding = embedding_factory.get_ollama_embedding("llama3.1")

        a = embedding.get_text_embedding(text1)
        b= embedding.get_text_embedding(text2)

        score = embedding.similarity(a,b)
        print(f"similarity score is: {score}")

        self.assertGreater(score, 0.75)

    def test_similarity_for_two_simple_words_with_huggingface_embedding(self):
        text1 = "dog"
        text2 = "puppy"
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding = embedding_factory.get_huggingface_embedding("BAAI/bge-base-en-v1.5")

        a = embedding.get_text_embedding(text1)
        b= embedding.get_text_embedding(text2)

        score = embedding.similarity(a,b)
        print(f"similarity score is: {score}")

        self.assertGreater(score, 0.75)

    def test_similarity_for_two_simple_sentences_with_huggingface_embedding(self):
        text1 = "dog is a canine."
        text2 = "a puppy is also a canine."
        embedding_factory = EmbeddingFactory("embedding-model")

        embedding = embedding_factory.get_huggingface_embedding("BAAI/bge-base-en-v1.5")

        a = embedding.get_text_embedding(text1)
        b= embedding.get_text_embedding(text2)

        score = embedding.similarity(a,b)
        print(f"similarity score is: {score}")

        self.assertGreater(score, 0.75)

if __name__ == '__main__':
    unittest.main()
