import unittest
from statistics import median


class MyTestCase(unittest.TestCase):
    def test_something(self):
        list = 101, 88, 90, 93, 100, 97, 99, 108, 115,107, 107, 93, 109, 128, 32,120, 123, 114, 109, 120, 54
        print(sum(list))
        print(median(list))
        print(sum(list)/len(list))


if __name__ == '__main__':
    unittest.main()
