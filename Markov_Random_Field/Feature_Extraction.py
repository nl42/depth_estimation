import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import tqdm,trange

import sys
sys.path.append('../')


from depth_Functions import (
    show_array_of_images, 
    show_img,
    stand
)

filters = [
                np.array([1, 4, 6, 4, 1]),
                np.array([-1, -2, 0, 2, 1]),
                np.array([-1, 0, 2, 0, -1]),
                np.array([1, -4, 6, -4, 1])
          ] 
def mask(patch, first, second=None):
    if second is None:
        return cv2.filter2D(patch, -1, filters[first].reshape(5,1)*filters[first])
    else:
        filter1 = cv2.filter2D(patch, -1, filters[first].reshape(5,1)*filters[second])
        filter2 = cv2.filter2D(patch, -1, filters[second].reshape(5,1)*filters[first])
        return (filter1 + filter2)/2

def texture_variation(patch_intensity):
    level, edge, spot, ripple = 0, 1, 2, 3
    masks =  [
                mask(patch_intensity, level, edge),
                mask(patch_intensity, level, ripple),
                mask(patch_intensity, edge, spot),
                mask(patch_intensity, spot),
                mask(patch_intensity, ripple),
                mask(patch_intensity, level, spot),
                mask(patch_intensity, edge),
                mask(patch_intensity, edge, ripple),
                mask(patch_intensity, spot, ripple),
             ]
    
    return masks

def haze(cr, cb):
    return [mask(cr, 0, 1), mask(cb, 0, 1)]

def create_kernels(step=90, min=0, max=180):
    kernels = []
    
    for angle in range(min,max,step):
        rad = math.radians(angle)
        cos = round(math.cos(rad),2)
        sin = round(math.sin(rad),2)
        kernels.append(np.array([
                                    [sin-cos,   2*sin, cos+sin],
                                    [-2*cos,        0,   2*cos],
                                    [-sin-cos, -2*sin, cos-sin] 
                                ]))
    
    return kernels

def denoise(patch, size=5, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g_kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return ndimage.filters.convolve(patch, g_kernel)

def texture_gradients(patch, *args, **kwargs):
    kernels = create_kernels(*args, **kwargs)
    gradients = []
    
    for kernel in kernels:
        gradients.append(ndimage.filters.convolve(patch, kernel))
        
    return gradients
    
# For if we implement thresholds:
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def get_thresholds(patch, sigma=0.33):
    v = np.median(patch)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 - sigma) * v))

    return lower, upper

# def get_features(patch, function x : np.sum(x)):
#     y, cr, cb = cv2.split(patch)
#     vector = []
#     vector = function(texture_variation(y))
#     vector += function(haze(cr, cb))
#     vector += function(texture_gradients(y, step=30))

#     return vector

# def create_feature_vector(patch, function):
#     vector = get_features(patch)

#     vector = [function(feature) for feature in vector]

#     return vector

# def sum_and_square_sums(vector):
#     vector = [(v,v**2) for v in vector]
#     return vector

# def feature_histogram(vector, bins=11):
#     return np.histogram(patch, bins=bins)[0]

def create_local_feature_vector(patch, squares=True, function=np.sum):
    y, cr, cb = cv2.split(patch)
    vector = []
    vector = texture_variation(y) 
    vector += haze(cr, cb)
    vector += texture_gradients(y, step=30)

    if not squares:
        return np.array([function(feature) for feature in vector])

    vector += [feature**2 for feature in vector]

    return np.array([function(feature) for feature in vector])


def get_feature_vector(image):
    y, cr, cb = cv2.split(image)
    
    vector = np.stack([*texture_variation(y),*haze(cr,cb),*texture_gradients(y, step=30)],axis=2)
        
    return vector

def get_features_with_squares(image):
    y, cr, cb = cv2.split(image)
    
    vector = np.stack([*texture_variation(y),*haze(cr,cb),*texture_gradients(y, step=30)],axis=2)

    return np.concatenate((vector, vector**2), axis=2)

def get_squared_features(image):
    y, cr, cb = cv2.split(image)
    
    vector = np.stack([*texture_variation(y),*haze(cr,cb),*texture_gradients(y, step=30)],axis=2)

    vector = vector**2    
    
    return vector
