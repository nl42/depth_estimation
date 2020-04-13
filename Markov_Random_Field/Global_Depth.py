import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import tqdm

import sys
sys.path.append('../')

from depth_Functions import (
    import_raw_colour_image, 
    show_img, 
    show_array_of_images
)
    
from Feature_Extraction import (
    process_patches, 
    create_local_feature_vector
)

def get_global_features(image, patchsize, relative_scales):
    #Process local features at scale 1
    global_features = process_patches(image, patchsize=patchsize, function=create_global_feature_vectors, name='local')

    # Process neighbouring features at scale 1
    process_patches(global_features, patchsize=(2*patchsize[0], 2*patchsize[1]), override=True, function=set_adjacent_features, name='neighbours')

    # Process columns at scale 1
    columnsize = (int(image.shape[0]/4),patchsize[1])
    column_features = process_patches(image, columnsize, function=create_local_feature_vector, name='columns')

    for x in range(image.shape[1]):
        global_features[:,x,-4:] = column_features[0:image.shape[0]:columnsize[0], x]
    
    # calculate the subpatches at scales
    process_scaled_features = lambda patch : subpatch_features(patch, relative_scales)
    scaled_features = process_patches(image, patchsize=patchsize, function=process_scaled_features, name='Scaled features')
    
    return global_features, scaled_features

def create_global_feature_vectors(patch):
    local_feature_vector = create_local_feature_vector(patch)
    return [local_feature_vector for i in range(9)]

def set_adjacent_features(patches):
    heights = [y for y in range(0, patches.shape[0]+1, int(patches.shape[0]/2))]
    widths = [x for x in range(0, patches.shape[1]+1, int(patches.shape[1]/2))]
    
    for y in range(0,2):
        for x in range(0,2):
            patches[heights[y]:heights[y+1], 
                    widths[x]:widths[x+1], 3-(2*y),:]   = patches[heights[1-y]:heights[2-y], 
                                                            widths[x]:widths[x+1], 0]
            patches[heights[y]:heights[y+1], 
                    widths[x]:widths[x+1], 2*(x+1),:]   = patches[heights[y]:heights[y+1], 
                                                              widths[1-x]:widths[2-x], 0]


def subpatch_features(patch, relative_scales):
    scaled_patch = cv2.resize(patch, (0,0), fx=relative_scales[0][0], fy=relative_scales[0][1])
    patchsize = (int(scaled_patch.shape[0]/3), int(scaled_patch.shape[1]/3))
    order = [(1,1),(0,1),(1,2),(2,1),(1,0)]
    
    subpatches = []
    
    for index, (y,x) in enumerate(order):
        subpatches.append(create_local_feature_vector(scaled_patch[y*patchsize[0]:(y+1)*patchsize[0], 
                                                                   x*patchsize[1]:(x+1)*patchsize[1]]))
    
    
    if len(relative_scales) > 2:
        return [subpatches, *subpatch_features(scaled_patch[y*patchsize[0]:(y+1)*patchsize[0], 
                                                           x*patchsize[1]:(x+1)*patchsize[1]], relative_scales[1:])]
    if len(relative_scales) == 2:
        return [subpatches, subpatch_features(scaled_patch[y*patchsize[0]:(y+1)*patchsize[0], 
                                                           x*patchsize[1]:(x+1)*patchsize[1]], relative_scales[1:])]
    
    return subpatches