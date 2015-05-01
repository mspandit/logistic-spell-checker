import cProfile

from spelling_checker import TrainingSet, SpellingChecker

def big():
    """docstring for big"""
    ts = TrainingSet()
    ts.set_file('big.txt')
    sc = SpellingChecker(ts)

if __name__ == "__main__":
    cProfile.run('big()')
