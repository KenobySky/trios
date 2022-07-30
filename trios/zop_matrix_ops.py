# cython: profile=True
# filename: wop_matrix_ops.pyx

from __future__ import print_function

import cv2
import numpy as np
from cv2 import imread

import trios.shortcuts.persistence as p


def count_pixels_mask(msk, win):
    hh = win.shape[0]
    ww = win.shape[1]
    hh2 = hh//2
    ww2 = ww//2
    h = msk.shape[0]
    w = msk.shape[1]
    
    count = 0

    for i in range(hh2, h-hh2):
        for j in range(ww2, w-ww2):
            if msk[i,j] != 0:
                count += 1
    
    return count


def process_image(dataset, win, iinput,  output, mask, extractor):
    h = iinput.shape[0]
    w = iinput.shape[1]
    z = iinput.shape[2]
    
    i=0
    j=0
    l=0
    m=0
    hh = win.shape[0]
    ww = win.shape[1]
    zz = win.shape[2]
    hh2 = hh/2
    ww2 = ww/2
    count = 0
    
    
    wpat = extractor.temp_feature_vector()
    for i in range(hh2, h-hh2):
        for j in range(ww2, w-ww2):
            if (not mask is None) and mask[i,j] > 0:
                count += 1
                
                extractor.extract(iinput, i, j, wpat)
                wpatt = tuple(wpat)
                if not wpatt in dataset:
                    dataset[wpatt] = {}
                if not output[i,j] in dataset[wpatt]:
                    dataset[wpatt][output[i,j]] = 1
                else:
                    dataset[wpatt][output[i,j]] += 1


def process_one_image(win,inp, msk, X, extractor, temp, k):
    h, w, i, j, l, ww, hh, ww2, hh2 = 0

    hh = win.shape[0]; ww = win.shape[1]
    hh2 = hh/2
    ww2 = ww/2
    h = inp.shape[0] ; w = inp.shape[1]
    for i in range(hh2, h-hh2):
        for j in range(ww2, w-ww2):
            if (not msk is None) and msk[i,j] > 0:
                extractor.extract(inp, i, j, temp)
                for l in range(temp.shape[0]):
                    X[k, l] = temp[l]
                k += 1 
    return k
    

def process_image_ordered(imageset, extractor):
    # count all pixels in mask
    npixels = 0
    k=0


    win = extractor.window
    
    hh = win.shape[0]
    ww = win.shape[1]
    zz = win.shape[2]
    hh2 = hh//2
    ww2 = ww//2
        
    for _i, _o, m in imageset:
        if m != None:
            msk = imread(m,cv2.IMREAD_GRAYSCALE)
        else:
            # if there is no mask we take the shape from the input image
            # and check for m != 0 below.
            msk = imread(_i, cv2.IMREAD_GRAYSCALE)

        for i in range(hh2, msk.shape[0]-hh2):
            for j in range(ww2, msk.shape[1]-ww2):
                if m is None or msk[i, j] != 0:
                    npixels += 1
                    
    temp = extractor.temp_feature_vector()
    X = np.zeros((npixels, len(extractor),zz), temp.dtype)
    y = np.zeros(npixels, np.uint8)
    k = 0
    for (inp, out, msk) in p.load_imageset(imageset, win,isGrayScale=False):
        msk_np = np.asarray(msk, dtype=np.uint8)
        out_np = np.asarray(out, dtype=np.uint8)
        idx_i, idx_j = np.nonzero(msk_np)
        y[k:k+idx_i.shape[0]] = out_np[idx_i,idx_j]
        extractor.extract_batch(inp, idx_i, idx_j, X[k:k+idx_i.shape[0]])
        k += idx_i.shape[0]
        
        
    return X, y


def  apply_loop(window, image,mask, classifier, extractor):
    h = image.shape[0]
    w = image.shape[1]
    wh = int(window.shape[0]/2)
    ww = int(window.shape[1]/2)
    output = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    pat = extractor.temp_feature_vector()
    count = 0
    for i in range(wh, h-wh):
        for j in range(ww, w-ww):
            if mask[i,j] > 0: 
                count += 1
                extractor.extract(image, i, j, pat)
                output[i,j] = classifier.apply(pat)
    
    return output
    

def compare_images(out, msk, res, x_border, y_border):

    w = out.shape[1]
    h = out.shape[0]

    nerrs = 0
    npixels = 0
    
    for i in range(y_border, h - y_border):
        for j in range(x_border, w - x_border):
            if msk[i,j] != 0:
                npixels += 1
                if out[i, j] != res[i, j]:
                    nerrs += 1
    
    return (nerrs, npixels)



def compare_images_binary(out,msk,res,x_border,y_border):

    w = out.shape[1]
    h = out.shape[0]

    TN, TP, FN, FP, npixels = 0
    TN = TP = FN = FP = 0
    
    for i in range(y_border, h - y_border):
        for j in range(x_border, w - x_border):
            if msk[i,j] != 0:
                npixels += 1
                if out[i, j] == 0 and res[i, j] == 0:
                    TN += 1
                elif out[i, j] == 0 and res[i, j] != 0:
                    FP += 1
                elif out[i, j] != 0 and res[i, j] == 0:
                    FN += 1
                elif out[i, j] != 0 and res[i, j] != 0:
                    TP += 1
    
    return (TP, TN, FP, FN, npixels)
