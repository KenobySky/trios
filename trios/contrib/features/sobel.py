
import numpy as np
from skimage import filters

from trios.feature_extractors.raw import RAWFeatureExtractor


class SobelExtractor(RAWFeatureExtractor):

    def extract_batch(self,  inp, idx_i, idx_j, X):
        mask = np.zeros_like(inp)
        mask[idx_i, idx_j] = 1
        edges = (255*filters.sobel(inp, mask)).astype('uint8')

        for l in range(idx_i.shape[0]):
            self.extract(edges, idx_i[l], idx_j[l], X[l])

    def write_state(self, obj_dict):
        RAWFeatureExtractor.write_state(self, obj_dict)

    def set_state(self, obj_dict):
        RAWFeatureExtractor.set_state(self, obj_dict)
