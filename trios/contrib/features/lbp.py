
import numpy as np
import skimage.morphology as sk

from trios.feature_extractors.base_extractor import FeatureExtractor
from trios.feature_extractors.raw import RAWFeatureExtractor


class LBPExtractor(RAWFeatureExtractor):

    def __init__(self, window=None, mul=1.0, **kw):
        new_window = window[1:-1, 1:-1]
        FeatureExtractor.__init__(self, new_window, **kw)
        self.dtype = np.uint8
        self.mul = mul

    def calculate_lbp(self, img, i, j):
        lbp = []
        H = img.shape[0]
        W = img.shape[1]
        if i > 0 and i < H-1 and j > 0 and j < W-1:
            lbp[:] = [ int(img[i - 1, j] > img[i,j]), int(img[i - 1, j + 1] > img[i,j]), int(img[i, j + 1] > img[i,j]), int(img[i + 1, j + 1] > img[i,j]), int(img[i + 1, j] > img[i,j]), int(img[i + 1, j - 1] > img[i,j]), int(img[i, j - 1] > img[i,j]), int(img[i - 1, j - 1] > img[i,j])]
        else: 
            return 0
        return int("".join(str(p) for p in lbp),2)


    def extract_batch(self, inp, idx_i, idx_j,  X):
        lbp_img = np.zeros_like(inp)
        mask = np.zeros_like(inp)
        mask[idx_i, idx_j] = 1
        mask = sk.dilation(mask, self.window)
        idx_im, idx_jm = np.nonzero(mask); 

        for l in range(idx_im.shape[0]):
            lbp_img[idx_im[l], idx_jm[l]] = self.calculate_lbp(inp, idx_im[l], idx_jm[l])
        for l in range(idx_i.shape[0]):
            self.extract(lbp_img, idx_i[l], idx_j[l], X[l])

    def write_state(self, obj_dict):
        FeatureExtractor.write_state(self, obj_dict)

    def set_state(self, obj_dict):
        FeatureExtractor.set_state(self, obj_dict)

