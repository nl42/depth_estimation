import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import trange

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

def create_local_feature_vector(patch, squares=True, function=np.sum):
    vector = []
    y, cr, cb = cv2.split(cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb))
    
    vector = texture_variation(y) 
    vector += haze(cr, cb)
    vector += texture_gradients(y, step=30)

    if not squares:
        return np.array([function(feature) for feature in vector])

    vector += [feature**2 for feature in vector]

    return np.array([function(feature) for feature in vector])

def get_patchsize(image_shape, patchsize):
    if isinstance(patchsize, int):
        patchsize = [patchsize,patchsize]
    else:
        patchsize = list(patchsize)

    while image_shape[0] % patchsize[0] > 0:
        patchsize[0] -= 1
        if image_shape[1] % patchsize[1] > 0:
            patchsize[1] -= 1
    
    while image_shape[1] % patchsize[1] > 0:
        patchsize[1] += 1
    
    return patchsize

def process_patches(image, patchsize, function=lambda x: x, name=''):
    patchsize = get_patchsize(image.shape, patchsize)

    nxm = [int(i_dim / p_dim) for i_dim, p_dim in zip(image.shape, patchsize)]
    
    # Currently we get the shape by quickly running the function on the whole image, should be improved
    patches = np.zeros((*nxm,*np.array(function(image)).shape))
    
    for y in trange(nxm[0], leave=False, desc=name):
        for x in range(nxm[1]):
            patches[y, x] = np.array(function(image[y*patchsize[0]:(y+1)*patchsize[0], x*patchsize[1]:(x+1)*patchsize[1]]))
    
    return patches

def image_from_patches(patches, patchsize):
    image = np.zeros((patches.shape[0]*patchsize[0], patches.shape[1]*patchsize[1], *patches.shape[2:]))
    
    patchsize = (int(image.shape[0] / patches.shape[0]), int(image.shape[1] / patches.shape[1]))
    
    for y in trange(patches.shape[0], leave=False):
        for x in range(patches.shape[1]):
            image[y*patchsize[0]:(y+1)*patchsize[0], x*patchsize[1]:(x+1)*patchsize[1]] = patches[y, x]
    
    return image


# else:
    #     heights , widths = zip*([(stride[0]*height, stride[1]*width)
    #                              for height, width in order])

    # if override:
    #     for y_0, y_1 in tqdm(zip(heights, heights[1:]), 
    #                         total=len(heights)-1, leave=False, desc=name):
    #         for x_0, x_1 in zip(widths, widths[1:]):
    #             function(image[y_0:y_1, x_0:x_1])
    # # elif override:
    # #     for y_0, y_1 in tqdm(zip(heights, heights[1:]), 
    # #                         total=len(heights)-1, leave=False, desc=name):
    # #         for x_0, x_1 in zip(widths, widths[1:]):
    # #             function(image[y_0:y_1, x_0:x_1],
    # #                      target[y_0:y_1, x_0:x_1])
