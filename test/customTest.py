import matplotlib.pyplot as plt
import numpy as np

import trios
from trios.feature_extractors import RAWFeatureExtractorRGB
from sys import getsizeof

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

    x,y = extractor(window=window, imageset_path="A:/MeusProjetosPython/KerasVesselSegmentation/DRIVE_input/custom_training.set", extractor=RAWFeatureExtractorRGB, to_float32=False)

    print(x.shape)
    # print(x[500].shape)

    # img = x[500].reshape((3,3,3))
    # plt.imshow(X=img)
    # plt.show()

    print(getsizeMB(window))
    print(getsizeMB(x))
    print(getsizeMB(y))


    # extractor = RAWFeatureExtractorRGB(window=window)
    #
    # image = img.imread("A:/MeusProjetosPython/KerasVessels/datasets/DRIVE/training/default/images/21_training.png")
    # print(image.shape)
    # print(type(image))
    #
    # pattern = np.zeros((window.shape[0] * window.shape[1], window.shape[2]))
    # extractor.extract(img=image, i=300, j=350, pattern=pattern)
    #
    # print(pattern)
    # print(pattern.shape)
    #
    # pattern = pattern.reshape((window.shape[0], window.shape[1], window.shape[2]))
    #
    # plt.imshow(X=pattern)
    # plt.show()


if __name__ == "__main__":
    main()
