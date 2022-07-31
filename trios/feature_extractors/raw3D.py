import math

import numpy as np

from trios.feature_extractors.base_extractor import FeatureExtractor
from trios.shortcuts.persistence import load_image
from trios.zop_matrix_ops import process_image, process_image_ordered


class RAWFeatureExtractorRGB(FeatureExtractor):

    def __init__(self, window=None, mul=1.0, **kw):
        FeatureExtractor.__init__(self, window, **kw)
        self.dtype = np.uint8
        self.mul = mul

    def __len__(self):
        return self.window.shape[0] * self.window.shape[1]

    def extract(self, img, i, j, pattern):

        win = self.window
        hh = win.shape[0]
        ww = win.shape[1]
        hh2 = math.floor(hh / 2)
        ww2 = math.floor(ww / 2)

        k = 0

        for L in range(-hh2, hh2 + 1):
            for M in range(-ww2, ww2 + 1):
                r = img[i + L, j + M, 0]
                g = img[i + L, j + M, 1]
                b = img[i + L, j + M, 2]

                if r != 0 and g != 0 and b != 0:
                    pattern[k, 0] = r
                    pattern[k, 1] = g
                    pattern[k, 2] = b

                    k += 1
        if self.mul != 1:
            for L in range(pattern.shape[0]):
                pattern[L, :] = (pattern[L, :] * self.mul)

    def write_state(self, obj_dict):
        FeatureExtractor.write_state(self, obj_dict)
        obj_dict['mul'] = self.mul

    def set_state(self, obj_dict):
        FeatureExtractor.set_state(self, obj_dict)
        self.mul = obj_dict['mul']

    def extract_dataset(self, imgset, ordered=False):
        '''
This method extracts patterns from an `trios.imageset.Imageset`. If `ordered=True`,
the resulting patterns will be a pair containing a matrix *X* of shape *(M, N)*, where
*M* is the number of pixels that are in the mask (if there is no mask, the sum of all
pixels in all images) and *N* is `len(self)`.

If `ordered=False`, the training set is return in `dict` format. The patterns are
stored in the keys as a tuple. Each pattern is associated with a `dict` in which the keys
are outputs pixel values and the values are the number of times that a certain output
co-ocurred with the pattern. See the example below. ::

    { (255, 0, 255): {0: 5, 255: 1},
      (255, 255, 255): {0: 3, 255: 10},
      ... }
        '''
        if ordered == True:
            return process_image_ordered(imgset, self)

        dataset = {}
        for (inp, out, msk) in imgset:
            inp = load_image(inp, grayscale=False)
            out = load_image(out, grayscale=True)
            if msk is not None:
                msk = load_image(msk, grayscale=True)
            else:
                msk = np.ones(inp.shape, inp.dtype)
            # TODO: assign 0 to all pixels in the border (depends on mask)
            process_image(dataset, self.window, inp, out, msk, self)
        return dataset

    def temp_feature_vector(self):
        return np.zeros(len(self), np.uint32)
