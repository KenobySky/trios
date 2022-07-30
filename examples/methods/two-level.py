import numpy as np
from sklearn.tree import DecisionTreeClassifier

import trios
import trios.shortcuts.window as w
from trios.classifiers import SKClassifier
from trios.feature_extractors import RAWFeatureExtractor, CombinationPattern

if __name__ == '__main__':
    np.random.seed(10) # set this to select the same window everytime
    images = trios.Imageset.read('../jung-images/level1.set')
    images2 = trios.Imageset.read('../jung-images/level2.set')
    test = trios.Imageset.read('../jung-images/test.set')

    domain = np.ones((9, 7), np.uint8)
    windows = [w.random_win(domain, 40, True) for i in range(5)]
    ops = []
    for i in range(5):
        op = trios.WOperator(windows[i], SKClassifier(DecisionTreeClassifier()), RAWFeatureExtractor)
        print('Training...', i)
        op.train(images)
        ops.append(op)
    
    comb = CombinationPattern(*ops)
    wop2 = trios.WOperator(comb.window, SKClassifier(DecisionTreeClassifier()), comb) 
    print('Training 2nd level')
    wop2.train(images2)
    
    print('Error', wop2.eval(test))

