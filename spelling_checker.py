import re, collections
import numpy
from sklearn.linear_model import LogisticRegression

def words(text): return re.findall('[a-z]+', text.lower())

def count_words(features):
    model = collections.defaultdict(lambda: 1)
    word_count = 0
    words_length = 0
    max_word_length = 0
    for f in features:
        model[f] += 1
        if (len(f) > max_word_length):
            max_word_length = len(f)
        word_count += 1
        words_length += len(f)
    return model, word_count, words_length, max_word_length

ALPHA_COUNT = 26
CHAR_COUNT = ALPHA_COUNT + 1 # spaces

class TrainingSet(object):
    """
    Generates word count, word length, and a mapping from words to counts from
    a corpus.
    """        
    def __init__(self):
        super(TrainingSet, self).__init__()

    def set_file(self, train_file='big.txt'):
        """docstring for set_train_file"""
        self.word2count, self.word_count, self.words_length, self.max_word_length = count_words(words(file(train_file).read()))

    def set_string(self, train_string):
        """docstring for set_train_string"""
        self.word2count, self.word_count, self.words_length, self.max_word_length = count_words(words(train_string))
        

class SpellingChecker(object):
    """docstring for SpellingChecker"""    
    def __init__(self, training_set):
        super(SpellingChecker, self).__init__()
        self.training_set = training_set
        self.average_word_length = int(training_set.words_length / training_set.word_count)
        self.classifier = LogisticRegression()
        training_inputs = [self.word2input(word) for word in self.training_set.word2count.keys()]
        training_outputs = [self.training_set.word2count.keys().index(word) for word in self.training_set.word2count.keys()]
        self.classifier.fit(training_inputs, training_outputs)

    def word2input(self, word):
        """docstring for word2input"""
        result = numpy.zeros((CHAR_COUNT * self.training_set.max_word_length))
        for i, char in enumerate(word[0:self.training_set.max_word_length]):
            result[i * CHAR_COUNT + "abcdefghijklmnopqrstuvwxyz".index(char)] = 1.0
        return result

    def word2output(self, word):
        """docstring for word2output"""
        result = numpy.zeros((len(self.training_set.word2count.keys())))
        result[self.training_set.word2count.keys().index(word)] = 1.0
        return result

    def check(self, word, n_best=3):
        """docstring for check"""
        result = self.classifier.predict(self.word2input(word))
        return self.training_set.word2count.keys()[result]