

import pickle
import gzip

import cv2
import numpy as np


def save_gzip(op, fname):
    with gzip.open(fname, 'wb') as f:
        pickle.dump(op, f, -1)


def load_gzip(fname):
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


def load_image(fname):
    return cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

def load_mask_image(fname, shape, win):
    if fname is not None and fname != '':
        msk = load_image(fname)
    else:
        msk = np.ones(shape, np.uint8)
    hh2 = win.shape[0] // 2
    ww2 = win.shape[1] // 2
    h, w = shape
    msk[:hh2] = msk[h-hh2:] = msk[:,:ww2] = msk[:, w-ww2:] = 0
    return msk


def save_image(image, fname):
    cv2.imwrite(fname, image)


def load_imageset(imgset, win):
    for (inp, out, msk) in imgset:
        inp = load_image(inp)
        out = load_image(out)
        msk = load_mask_image(msk, inp.shape, win)
        yield (inp, out, msk)


