import numpy as np
from sklearn.tree import DecisionTreeClassifier

import trios
from trios.classifiers import SKClassifier
from trios.contrib.features.fourier import FourierExtractor

drive_location = 'datasets/drive'
training = trios.Imageset([
    ('%s/training/images/%2d_training.tif'%(drive_location, i),
    '%s/training/1st_manual/%2d_manual1.gif'%(drive_location, i),
    '%s/training/mask/%2d_training_mask.gif'%(drive_location, i))
    for i in range(21, 41)])

testset = trios.Imageset([
    ('%s/test/images/%02d_test.tif'%(drive_location, i),
    '%s/test/1st_manual/%02d_manual1.gif'%(drive_location, i),
    '%s/test/mask/%02d_test_mask.gif'%(drive_location, i))
    for i in range(1, 21)])

if __name__ == '__main__':
   win = np.ones((9,9), np.uint8)
   op = trios.WOperator(win, SKClassifier(DecisionTreeClassifier()), FourierExtractor)
   print('Training')
   op.train(training)
   print('Evaluating')
   print('Accuracy:', 1 - op.eval(testset))
