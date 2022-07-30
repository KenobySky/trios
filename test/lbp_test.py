import numpy as np
from sklearn.tree import DecisionTreeClassifier

import trios
import trios.shortcuts.persistence as p
from trios.classifiers import SKClassifier
from trios.contrib.features.lbp import LBPExtractor

if __name__ == '__main__':
    images = trios.Imageset.read('images/training.set')
    win = np.ones((5, 5), np.uint8)
    
    op = trios.WOperator(win, SKClassifier(DecisionTreeClassifier()), LBPExtractor)
    op.train(images)

    img= p.load_image('images/jung-1a.png')
    lbp = op.apply(img, img)

    test = trios.Imageset.read('images/test.set')
    print('Accuracy', op.eval(test, procs=7))

    print(op.cite_me())
