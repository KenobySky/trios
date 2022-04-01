import math

import numpy as np

from trios.feature_extractors.base_extractor import FeatureExtractor


class RAWFeatureExtractor(FeatureExtractor):

    def __init__(self,  window=None, mul=1.0, **kw):
        FeatureExtractor.__init__(self, window, **kw)
        self.dtype = np.uint8
        self.mul = mul

    def __len__(self):
        return np.greater(self.window, 0).sum()

    def extract(self, img, i,  j, pattern):

        win = self.window
        hh = win.shape[0]
        ww = win.shape[1]
        hh2 = math.floor(hh/2)
        ww2 = math.floor(ww/2)

        k = 0

        #print(type(pattern.shape))
        #print(pattern.shape)

        for l in range(-hh2, hh2+1):
            for m in range(-ww2, ww2+1):
                if win[l+hh2, m+ww2] != 0:

                    aux = img[i+l, j+m]

                    pattern[k] = aux
                    k += 1
        if self.mul != 1:
            for l in range(pattern.shape[0]):
                pattern[l] = (pattern[l] *self. mul)

    def write_state(self, obj_dict):
        FeatureExtractor.write_state(self, obj_dict)
        obj_dict['mul'] = self.mul

    def set_state(self, obj_dict):
        FeatureExtractor.set_state(self, obj_dict)
        self.mul = obj_dict['mul']


def RAWBitExtract( win, img,  i,  j,  pattern):
    hh = win.shape[0]
    ww = win.shape[1]
    hh2 = math.floor(hh/2)
    ww2 = math.floor(ww/2)
    l, m, shift, byt = 0
    k = 0


    for l in range(pattern.shape[0]):
        pattern[l] = 0

    for l in range(-hh2, hh2+1):
        for m in range(-ww2, ww2+1):
            if win[l+hh2, m+ww2] != 0:
                shift = k / 32
                byt = k % 32
                if img[i+l,j+m] != 0:
                    pattern[shift] = pattern[shift] | (1 << byt)
                k += 1

class RAWBitFeatureExtractor(FeatureExtractor):
    def __init__(self, window=None):
        FeatureExtractor.__init__(self, window, dtype=np.uint32)

    def __len__(self):
        wsize = np.greater(self.window, 0).sum()
        if wsize % 32 == 0:
            return int(wsize / 32)
        else:
            return int(wsize / 32) + 1

    def temp_feature_vector(self):
        return np.zeros(len(self), np.uint32)

    def extract(self,  img,  i, j,  pattern):
        RAWBitExtract(self.window, img, i, j, pattern)

