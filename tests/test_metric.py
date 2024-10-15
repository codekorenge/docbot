import unittest
from fileinput import filename
from statistics import median

from nltk import word_tokenize


class MyTestCase(unittest.TestCase):
    def test_something(self):
        list = 101, 88, 90, 93, 100, 97, 99, 108, 115,107, 107, 93, 109, 128, 32,120, 123, 114, 109, 120, 54
        print(sum(list))
        print(median(list))
        print(sum(list)/len(list))

    def test_tokens(self):
        text = open("../sample_prompts/sample1.txt", "r")
        cleaned_text = ""
        for line in text:
            line = line.rstrip()
            cleaned_text += line
            print(line)
        tokens = word_tokenize(cleaned_text)
        count = len(tokens)
        print(tokens)


if __name__ == '__main__':
    unittest.main()
