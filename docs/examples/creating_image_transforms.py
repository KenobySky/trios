from sklearn.tree import DecisionTreeClassifier

import trios
import trios.shortcuts.persistence as p
import trios.shortcuts.window as trios_win
from trios.classifiers import SKClassifier
from trios.feature_extractors import RAWFeatureExtractor

if __name__ == '__main__':
    training_set = trios.Imageset.read('jung-images/training.set')
    test = trios.Imageset.read('jung-images/test.set')
        
    win = trios_win.rectangle(9, 7)
    classifier = SKClassifier(DecisionTreeClassifier())
    fext = RAWFeatureExtractor(win)
    op = trios.WOperator(win, classifier, fext)
    
    op.train(training_set)
    p.save_gzip(op, 'basic-jung.op.gz')
    
    print('Error: ', op.eval(test))
