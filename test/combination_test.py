import numpy as np
from sklearn import svm

import trios
from trios.classifiers import SKClassifier
from trios.contrib.features.featurecombination import FeatureCombinationExtractor
from trios.contrib.features.lbp import LBPExtractor
from trios.feature_extractors import RAWFeatureExtractor

if __name__ == '__main__':
    images = trios.Imageset.read('../images/training.set')
    win = np.ones((5, 5), np.uint8)

    lbp = LBPExtractor(window=win, batch_size=1000)
    raw = RAWFeatureExtractor(window=win, batch_size=1000)
    feats = []
    feats.append(lbp)
    feats.append(raw)
    combination = FeatureCombinationExtractor(*feats)
    op = trios.WOperator(win, SKClassifier(svm.LinearSVC(class_weight='balanced'), partial=True), combination)
    op.train(images)

    test = trios.Imageset.read('../images/test.set')
    print('Accuracy', op.eval(test, procs=1))
