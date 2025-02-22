import numpy as np
from sklearn.tree import DecisionTreeClassifier

import trios
from trios.classifiers import SKClassifier
from trios.contrib.features.lbp import LBPExtractor

chars_location = 'datasets/'
training = trios.Imageset.read(chars_location + 'characters/training.set')
testset = trios.Imageset.read(chars_location + 'characters/test.set')

if __name__ == "__main__":
    win = np.ones((9,9), np.uint8)
    op = trios.WOperator(win, SKClassifier(DecisionTreeClassifier()), LBPExtractor)
    print('Training')
    op.train(training)
    print('Evaluating')
    print('Accuracy:', 1 - op.eval(testset))
