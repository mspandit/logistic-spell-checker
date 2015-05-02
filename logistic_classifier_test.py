import unittest
from logistic_classifier import LogisticClassifier, sigmoid_gradient

class LogisticClassifierTest(unittest.TestCase):
    """docstring for LogisticClassifierTest"""

    def test_instantiation(self):
        """docstring for test_instantiation"""
        lc = LogisticClassifier(5, 4)

    def test_cost_calculation(self):
        """docstring for test_cost_calculation"""
        lc = LogisticClassifier(5, 4)
        output = lc.output_activation([0, 1, 0, 1, 0])
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 4)

    def test_sigmoid_gradient(self):
        """docstring for test_sigmoid_gradient"""
        self.assertTrue(sigmoid_gradient(100) < 1.0e-43)
        self.assertTrue(sigmoid_gradient(-100) < 1.0e-43)
        self.assertEqual(0.25, sigmoid_gradient(0))

    def test_train(self):
        """docstring for test_backpropagation"""
        lc = LogisticClassifier(5, 4)
        for i in xrange(100):
            lc.train([[[0, 1, 0, 1, 0]]], [[1, 0, 1, 0]])
        activation = lc.output_activation([0, 1, 0, 1, 0])
        self.assertTrue(activation[0][0] > 0.990)
        self.assertTrue(activation[0][1] < 0.01)
        self.assertTrue(activation[0][2] > 0.990)
        self.assertTrue(activation[0][3] < 0.01)
