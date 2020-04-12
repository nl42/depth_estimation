import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import tqdm

import sys
sys.path.append('../')

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
    
    masks = [
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
    gausian = denoise(patch)
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

def create_local_feature_vector(patch):
    vector = []
    y, cr, cb = cv2.split(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb))
    
    vector = texture_variation(y) 
    vector += haze(cr, cb)
    vector += texture_gradients(y, step=30)
    vector += [feature**2 for feature in vector]

    return [np.sum(feature) for feature in vector]

def process_patches(image, patchsize, order=None, override=False, function=lambda x: x, stride=None):
    if stride is None:
        stride = patchsize
    
    if order = None:
        heights = [y for y in range(0, image.shape[0]+1, stride[0])]
        widths = [x for x in range(0, image.shape[1]+1, stride[1])]
    else:
        heights , widths = zip*([(stride[0]*height, stride[1]*width)
                                 for height, width in order])

    if override:
        for y_0, y_1 in tqdm(zip(heights, heights[1:]), 
                            total=len(heights)-1, leave=False):
            for x_0, x_1 in zip(widths, widths[1:]):
                function(image[y_0:y_1, x_0:x_1])
    else:
        target = np.zeros((*image.shape[0:2],*np.array(function(image)).shape))

        for y_0, y_1 in tqdm(zip(heights, heights[1:]), 
                            total=len(heights)-1, leave=False):
            for x_0, x_1 in zip(widths, widths[1:]):
                target[y_0:y_1, x_0:x_1] = np.array(function(image[y_0:y_1, x_0:x_1]))
    
        return target

def nxm_patches(image, nxm, *args, **kwargs):
    height = int(image.shape[0] / nxm[0])
    width = int(image.shape[1] / nxm[1])
    
    return process_patches(image, (height, width), *args, **kwargs)