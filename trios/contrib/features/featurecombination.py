
import numpy as np

from trios.feature_extractors.base_extractor import FeatureExtractor


class FeatureCombinationExtractor(FeatureExtractor):

    #def public list features

    def __init__(self, *features, **kwargs):
        self.features = list(features)
        win = self.features[0].window
        FeatureExtractor.__init__(self, win, **kwargs)
        self.dtype=np.uint8

    def __len__(self):
        n = 0
        for i in range(len(self.features)):
            n += len(self.features[i])
        return n

    def extract_batch(self,  inp, idx_i, idx_j,  X):
        begin = 0
        for i in range(len(self.features)):
            n = len(self.features[i])
            self.features[i].extract_batch(inp, idx_i, idx_j, X[:,begin:begin+n])
            begin += n

    def write_state(self, obj_dict):
        FeatureExtractor.write_state(self, obj_dict)
        obj_dict['features'] = self.features

    def set_state(self, obj_dict):
        FeatureExtractor.set_state(self, obj_dict)
        self.features = obj_dict['features']
