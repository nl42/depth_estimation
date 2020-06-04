import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import trange, tqdm

import sys
sys.path.append('../')

from depth_Functions import (
    import_raw_colour_image, 
    show_img, 
    show_array_of_images
)
    
from Feature_Extraction import (
    create_local_feature_vector
)

from Patches import Patches

def get_absolute_and_relative_depth(image, patchsize, scales):
    patchsize = get_patchsize(image.shape, patchsize)
    scale_1, larger_scales = get_absolute_depth(image, patchsize, scales)
    absolute_depth = np.concatenate((scale_1, *larger_scales),2).reshape(*scale_1.shape[:2], -1)
    
    relative_histogram = get_relative_histograms(image, patchsize, scales)
    
    return absolute_depth, relative_histogram

def get_relative_histograms(image, patchsize, scales):
    patchsize = get_patchsize(image.shape, patchsize)
    
    histogram_10 = lambda feature : np.histogram(feature, bins=10)[0]
    local_hist = lambda  patch : create_local_feature_vector(patch, squares=False, function=histogram_10)
    global_hist = lambda  patch : create_feature_vector_with_neighbours(patch, squares=False, function=histogram_10)    
    
    relative_depths = [process_patches(image, patchsize, function=global_hist)]
    relative_depths[0] = subtract_adjacent(relative_depths[0])
    
    relative_depths += get_scaled_features(image, patchsize, scales, local_hist, sub=True) 
    
    return relative_depths

def get_absolute_depth(image, patchsize, scales):
    patchsize = get_patchsize(image.shape, patchsize)
    
    #Process local features at scale 1
    global_features = process_patches(image, patchsize, function=create_global_feature_vectors, name='local')

    # Process neighbouring features at scale 1
    set_adjacent(global_features)

    # Process columns at scale 1
    columnsize = (int(image.shape[0]/4),patchsize[1])
    column_features = process_patches(image, columnsize, function=create_local_feature_vector, name='columns')

    for x in range(column_features.shape[1]):
        global_features[:,x,-4:] = column_features[:, x]
    
    # calculate the subpatches at scales
    scaled_features = get_scaled_features(image, patchsize, scales)
    
    return global_features, scaled_features

def get_scaled_features(image, patchsize, scales, function=create_local_feature_vector, sub=False):
    scaled_features = []

    for (dy,dx) in tqdm(scales, total=len(scales), desc='Scales'):
        scaled_image = cv2.resize(image, (0,0), fx=dx, fy=dy)
        scaled_patchsize = (patchsize[0] * dy, patchsize[1] * dx)
        scaled_features.append(process_patches(scaled_image, patchsize=scaled_patchsize, 
                        function=lambda patch : scaled_patch(patch, patchsize, (dy,dx), function, sub), name=f'{dy}x{dx}'))
    
    return scaled_features

def scaled_patch(patch, patchsize, scale, function, sub=False):
    subpatches = []
    centre = (int(scale[0]/2)*patchsize[0], int(scale[1]/2)*patchsize[1])
    order = [(-1,0),(0,1),(1,0),(0,-1)]
    centre_patch = function(patch[centre[0]:centre[0]+patchsize[0], centre[1]:centre[1]+patchsize[1]])
    
    if not sub:
        subpatches[0] = centre_patch

    for (y,x) in order:
        start = (centre[0]+(y*patchsize[0]), centre[1]+(x*patchsize[1]))
        end = (start[0]+patchsize[0], start[1]+patchsize[1])
        if sub:
            subpatches.append(centre_patch - function(patch[start[0]:end[0], start[1]:end[1]]))
        else :
            subpatches.append(function(patch[start[0]:end[0], start[1]:end[1]]))
    return subpatches