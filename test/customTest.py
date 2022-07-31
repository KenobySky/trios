from sys import getsizeof

import numpy as np
import matplotlib.pyplot as plt
import trios
from trios.feature_extractors import RAWFeatureExtractorRGB


def getsizeMB(obj):
    size = round(getsizeof(obj) / 1024 / 1024, 2)
    return size


def extractor(window, imageset_path, extractor, to_float32=True):
    imageset = trios.Imageset.read(imageset_path)
    x, y = extractor(window).extract_dataset(imageset, True)

    if to_float32 == True:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
    return x, y


def main():
    window = np.ones(shape=(31, 31, 3), dtype=np.uint8)

    x,y = extractor(window=window, imageset_path="A:/MeusProjetosPython/KerasVesselSegmentation/DRIVE_input/smallset.set", extractor=RAWFeatureExtractorRGB, to_float32=False)

    print(x.shape)
    print(getsizeMB(window))
    print(getsizeMB(x))
    print(getsizeMB(y))

    k = x[1000].reshape((31,31,3))
    plt.imshow(k)
    plt.show()



if __name__ == "__main__":
    main()
