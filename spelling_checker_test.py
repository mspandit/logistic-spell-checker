import unittest
from spelling_checker import SpellingChecker, TrainingSet

class SpellingCheckerTest(unittest.TestCase):
    """docstring for SpellingCheckerTest"""

    def test_instantiation(self):
        """docstring for test_instantiation"""
        ts = TrainingSet()

    def test_correction(self):
        """docstring for test_correction"""
        ts = TrainingSet()
        ts.set_string('peas porridge hot peas porridge cold peas porridge in the pot nine days old')
        sc = SpellingChecker(ts)
        self.assertEqual('peas', sc.check('peas'))
        self.assertEqual('peas', sc.check('pxas'))
        self.assertEqual('porridge', sc.check('parridge'))

    def test_large_corpus(self):
        """docstring for test_large_corpus"""
        ts = TrainingSet()
        ts.set_file('big.txt')
        sc = SpellingChecker(ts)
