import cv2
import trios
from trios.feature_extractors import RAWFeatureExtractor
import numpy as np

def extractor(window, imageset_path, extractor,  to_float32=True):
    imageset = trios.Imageset.read(imageset_path)
    x, y = extractor(window).extract_dataset(imageset, True)

    if to_float32 == True:
        x = x.astype(np.float32)
        y = y.astype(np.float32)
    return x, y

def main():
    window = np.ones(shape=(3, 3), dtype=np.uint8)
    x, y = extractor(window=window,
                     imageset_path="A:/MeusProjetosPython/KerasVesselSegmentation/DRIVE_input/smallset.set",
                     extractor=RAWFeatureExtractor)
    cv2.imwrite("temp.jpg", x[0].reshape(3,3))


if __name__ == "__main__":
    main()
