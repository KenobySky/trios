import numpy as np
from sklearn.svm import LinearSVC

import trios
from trios.classifiers import SKClassifier
from trios.contrib.kern_approx import NystromFeatures
from trios.feature_extractors import RAWFeatureExtractor

if __name__ == '__main__':
    np.random.seed(10) 
    images = trios.Imageset.read('../jung-images/training.set')
    test = trios.Imageset.read('../jung-images/test.set')

    domain = np.ones((9, 7), np.uint8)
    raw = RAWFeatureExtractor(domain)
    ka_features = NystromFeatures(raw, images, n_components=1000, 
        kernel='poly', degree=3, batch_size=20000)
    svm = SKClassifier(LinearSVC(), partial=True)
    op = trios.WOperator(domain, svm, ka_features)
    op.train(images)

    print('Error:', op.eval(test))
