import math
import numpy

def sigmoid(x):
    return 1.0 / (1 + math.e ** -x)

def sigmoid_gradient(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def activation(weights, inputs):
    """
    weights: rows = output classes; 
    columns = input features + bias
    
    inputs: rows = input examples;
    columns = input features + bias
    
    return value: rows = output for each input;
    columns = probability of each output class
    """
    return sigmoid(weights.dot(inputs.T)).T

def cost(inputs, actuals, requireds):
    """
    inputs: rows = input examples;
    columns = input features + bias
    
    actuals: rows = actual outputs;
    columns = probability of each output class
    
    requireds: rows = required outputs;
    columns = probability of each output class
    
    return value: scalar cost
    """
    outer_sum = 0.0
    for example in zip(inputs, actuals, requireds):
        inner_sum = 0.0
        for classes in zip(example[1][0], example[2]):
            inner_sum += (
                - classes[1] * math.log(classes[0]) 
                - (1 - classes[1]) * math.log(1 - classes[0]))
        outer_sum += inner_sum
    return 1.0 / inputs.shape[0] * outer_sum

class LogisticClassifier(object):
    """docstring for LogisticClassifier"""
    def __init__(self, num_input_features, num_output_classes, learning_rate=1.0, epsilon=0.12):
        super(LogisticClassifier, self).__init__()
        # 1 + for bias
        self.weights = numpy.random.rand(
            num_output_classes, 
            1 + num_input_features) * 2 * epsilon - epsilon
        self.learning_rate = learning_rate

    def biased(self, activation):
        """
        activation: row vector
        
        return value: row vector
        """
        return numpy.append(numpy.ones((1, 1)), numpy.array(activation))

    def output_activation(self, input_activation):
        """
        input_activation: row vector of (unbiased) input features
        
        return value: row vector probability of each output class
        """
        return activation(
            self.weights, 
            self.biased(input_activation).reshape(1, self.weights.shape[1])) 

    def cost(self, inputs, requireds):
        """docstring for cost"""
        inputs = numpy.array(inputs)
        requireds = numpy.array(requireds)
        if inputs.shape[0] != requireds.shape[0]:
            raise "Shape mismatch"
        output_activations = []
        for inp in inputs:
            output_activations.append(self.output_activation(inp))
        return cost(inputs, output_activations, requireds)

    def cost_gradient(self, inputs, requireds):
        """docstring for cost_gradient"""
        delta = numpy.zeros(self.weights.shape)
    
        for example in zip(inputs, requireds):
            x = numpy.array(example[0])
            y = numpy.array(example[1])
            d = (self.output_activation(x) - y)
            delta += numpy.append(
                numpy.zeros((self.weights.shape[0], 1)), 
                x.T.dot(d).T, 
                axis=1)
    
        delta /= len(inputs)
        return delta

    def train(self, inputs, requireds):
        """docstring for train"""
        self.weights -= self.learning_rate * self.cost_gradient(inputs, requireds)
