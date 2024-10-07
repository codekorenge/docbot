import unittest

from helpers.llama_helper import get_text_embedding, get_ollama_embedding


class TestSemantic(unittest.TestCase):
    def test_similarity_for_two_simple_words(self):
        text1 = "dog"
        text2 = "puppy"

        embedding = get_ollama_embedding("llama3.1")

        a = get_text_embedding(embedding,text1)
        b= get_text_embedding(embedding,text2)

        score = embedding.similarity(a,b)
        print(f"similarity score is: {score}")

        self.assertGreater(score, 0.75)

    def test_similarity_for_two_simple_sentences(self):
        text1 = "dog is a canine."
        text2 = "a puppy is also a canine."

        embedding = get_ollama_embedding("llama3.1")

        a = get_text_embedding(embedding,text1)
        b= get_text_embedding(embedding,text2)

        score = embedding.similarity(a,b)
        print(f"similarity score is: {score}")

        self.assertGreater(score, 0.75)

if __name__ == '__main__':
    unittest.main()
