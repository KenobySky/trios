import gzip
import pickle

import cv2 as cv2
import numpy as np


def save_gzip(op, fname):
    with gzip.open(fname, 'wb') as f:
        pickle.dump(op, f, -1)


def load_gzip(fname):
    with gzip.open(fname, 'rb') as f:
        return pickle.load(f)


def load_image(fname, grayscale=True):
    img = None
    if grayscale:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def load_mask_image(fname, shape, win):
    if fname is not None and fname != '':
        msk = load_image(fname)
    else:
        msk = np.ones(shape, np.uint8)
    hh2 = win.shape[0] // 2
    ww2 = win.shape[1] // 2
    h, w, zz = shape
    msk[:hh2] = msk[h - hh2:] = msk[:, :ww2] = msk[:, w - ww2:] = 0
    return msk


def save_image(image, fname):
    cv2.imwrite(fname, image)


def load_imageset(imgset, win, isGrayScale=True):
    for (inp, out, msk) in imgset:
        inp = load_image(inp, isGrayScale)
        out = load_image(out)
        msk = load_mask_image(msk, inp.shape, win)
        yield (inp, out, msk)
