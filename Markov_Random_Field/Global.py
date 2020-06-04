import cv2
import numpy as np
import math
from tqdm.notebook import tqdm,trange
from itertools import product, chain, accumulate, zip_longest
from functools import partial
from operator import mul as multiply

from IPython.core.debugger import Tracer

from depth_Functions import (
    show_array_of_images,
    show_img,
    stand
)

from Feature_Extraction import calculate_local_features

def calc_features(image, local_function, neighbours=True, patch_function=np.sum, relative_scales=None):
    features = local_function(image)
    if neighbours:
        features = calc_neighbours(features)
    if relative_scales is not None:
        features = calc_scales(features, patch_function, relative_scales)
    
    return features

def unravel_patches(patches, shape):
    
    if patches.shape == shape:
        return patches
    
    patchshape = (math.ceil(shape[0]/patches.shape[0]), math.ceil(shape[1]/patches.shape[1]))
    patches = np.repeat(patches, patchshape[0], axis=0)
    patches = np.repeat(patches, patchshape[1], axis=1)
    y_shift = patches.shape[0] - shape[0]
    x_shift = patches.shape[1] - shape[1]
    x_pad = (x_shift//2, math.ceil(x_shift/2))
    # Tracer()()
    return patches[y_shift:, x_pad[0]:patches.shape[1]-x_pad[1]]

def patch_pad(image, patchshape):
    y_modulus = image.shape[0]%patchshape[0]
    x_modulus = image.shape[1]%patchshape[1]
    
    if y_modulus==x_modulus==0:
        return image

    y_shift = 0 if y_modulus==0 else patchshape[0]-y_modulus
    x_shift = 0 if x_modulus==0 else patchshape[1]-x_modulus
    y_pad = (y_shift,0)
    x_pad = (x_shift//2, math.ceil(x_shift/2))
    # print(y_pad, x_pad)

    empty = [(0,0) for i in range(len(image.shape)-2)]
    return np.pad(image, (y_pad, x_pad, *empty), mode='reflect')

def calc_patches(image, patch_function, patchshape):
    patches = patch_pad(image, patchshape)
    patches = patches.reshape(patches.shape[0]//patchshape[0], patchshape[0], patches.shape[1]//patchshape[1], patchshape[1], *patches.shape[2:])
    patches = np.swapaxes(patches, 1,2)
    if patch_function is not None:
        patches = patch_function(patches, axis=(2,3))
    return patches

def calc_scales(image, neighbours, patch_function, relative_scales):
    t_mul = lambda *tuples : np.product(tuples,axis=0) 
    scales = list(accumulate(relative_scales, t_mul))
    scaled_images = [calc_patches(image, patch_function, scale) for scale in scales]
    if neighbours:
        scaled_images = [calc_neighbours(image) for image in scaled_images]
    return scaled_images

def calc_relative_histograms(patches, bins, axis, *kwargs):
    histograms = calc_histograms(patches, bins, axis, *kwargs)
    return calc_relative(histograms)

def calc_bins(image, bin_number=10):
    return np.array([np.histogram_bin_edges(image[...,f], bins=bin_number) for f in range(image.shape[-1])])

def calc_histograms(patches, bins, axis):
    bins = bins[:,:-1]
    digitised = np.stack([np.digitize(patches[...,f], bins[f], right=False) for f in range(bins.shape[0])],axis=-1)

    return np.sum(np.identity(bins.shape[-1])[digitised-1], axis=axis)

# def calc_histograms(patches, axis, bin_number=10):
#     bins = np.array([np.histogram_bin_edges(patches[...,f], bins=bin_number) for f in range(patches.shape[-1])])
    
#     histogram = [np.histogram(patches[y,:,x,:,f], bins=bins[f])[0]
#                  for (y,x,f) in tqdm(product(range(patches.shape[0]), range(patches.shape[2]), range(patches.shape[-1])), total=patches.shape[0]*patches.shape[2]*patches.shape[-1], leave=False, desc=f'({patches.shape[1]}x{patches.shape[3]})')]
                
#     return (np.array(histogram).reshape(patches.shape[0], patches.shape[2], patches.shape[-1], bin_number), bins)


def calc_relative(image, n=1):
    empty = [(0,0) for i in range(len(image.shape)-2)]
    up    = np.pad(np.diff(image, n=n, axis=0),                 ((1,0),(0,0),*empty), mode='constant')
    right = np.pad(np.diff(image[:,::-1], n=n, axis=1)[:,::-1], ((0,0),(0,1),*empty), mode='constant')
    down  = np.pad(np.diff(image[::-1], n=n, axis=0)[::-1],     ((0,1),(0,0),*empty), mode='constant')
    left  = np.pad(np.diff(image, n=n, axis=1),                 ((0,0),(1,0),*empty), mode='constant')    
    return np.stack([up, right, down, left], axis=2)


def calc_neighbours(image, n=1):
    tuples = np.array([(0,0) for i in range(len(image.shape))])
    up    = np.pad(image[:-n],  ((1,0),*tuples[1:]), mode='edge')
    right = np.pad(image[:,n:], ((0,0),(0,1),*tuples[2:]), mode='edge')
    down  = np.pad(image[n:],   ((0,1),*tuples[1:]), mode='edge')
    left  = np.pad(image[:,:-n],((0,0),(1,0),*tuples[2:]), mode='edge')    
    return np.stack([image, up, right, down, left], axis=-2)