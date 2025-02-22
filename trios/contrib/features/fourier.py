import numpy as np
import numpy.fft as fft

from trios.feature_extractors.base_extractor import FeatureExtractor


class FourierExtractor(FeatureExtractor):

    def __init__(self, window=None, **kw):
        FeatureExtractor.__init__(self,window, **kw)
        self.dtype = np.float

    def __len__(self):
        begin = np.transpose(np.nonzero(self.window))[0]
        end = np.transpose(np.nonzero(self.window))[-1]
        n = np.ceil((end - begin + 1)/2)
        return int(n[0]*n[1])


    def extract(self, img, i, j, pat):
        pattern = pat
        win = self.window
        hh = win.shape[0]
        ww = win.shape[1]
        hh2 = hh/2
        ww2 = ww/2
        k = 0
        mask = np.zeros_like(win)

        for l in range(-hh2, hh2+1):
            for m in range(-ww2, ww2+1):
                mask[l,m] = img[i+l,j+m] * win[l+hh2,m+ww2]
        begin = np.transpose(np.nonzero(win))[0]
        end = np.transpose(np.nonzero(win))[-1]
        fourier = fft.fft2(mask[begin[0]:end[0],begin[1]:end[1]])
        fourier = np.absolute(fourier)

        for l in range(np.ceil(fourier.shape[0]/2)):
            for m in range(np.ceil(fourier.shape[1]/2)):
                if win[l,m] != 0:
                    pattern[k] = fourier[l,m]
                    k += 1

    def write_state(self, obj_dict):
        FeatureExtractor.write_state(self, obj_dict)

    def set_state(self, obj_dict):
        FeatureExtractor.set_state(self, obj_dict)
