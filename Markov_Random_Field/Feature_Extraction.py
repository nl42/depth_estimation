import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import tqdm,trange
from depth_Functions import stand

from IPython.core.debugger import Tracer

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

titles = [
                'Laws mask 1: Intensity',
                'Laws mask 2: Intensity',
                'Laws mask 3: Intensity',
                'Laws mask 4: Intensity',
                'Laws mask 5: Intensity',
                'Laws mask 6: Intensity',
                'Laws mask 7: Intensity',
                'Laws mask 8: Intensity',
                'Laws mask 9: Intensity',
                'Laws mask 1: Relative Red',
                'Laws mask 1: Relative Blue',
                'Gradient: 0',
                'Gradient: 30',
                'Gradient: 60',
                'Gradient: 90',
                'Gradient: 120',
                'Gradient: 150'
]

def calculate_local_features(image, convert=None, blur=True, squares=False, std=True):
    if convert is not None:
        image = cv2.cvtColor(image, convert)
    if blur:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    y, cr, cb = cv2.split(image)
    
    feature_array = np.concatenate([texture_variation(y),haze(cr,cb),texture_gradients(y, 30)],axis=2)
    
    if squares:
        feature_array = np.concatenate((feature_array, feature_array**2), axis=2)

    if std:
        feature_array = stand(feature_array)

    return feature_array

def mask(image, first, second=None):
    if second is None:
        return cv2.filter2D(image, -1, filters[first].reshape(5,1)*filters[first])
    else:
        filter1 = cv2.filter2D(image, -1, filters[first].reshape(5,1)*filters[second])
        filter2 = cv2.filter2D(image, -1, filters[second].reshape(5,1)*filters[first])
        return (filter1 + filter2)/2

def texture_variation(image_intensity):
    level, edge, spot, ripple = 0, 1, 2, 3
    masks =  [
                mask(image_intensity, level, edge),
                mask(image_intensity, level, ripple),
                mask(image_intensity, edge, spot),
                mask(image_intensity, spot),
                mask(image_intensity, ripple),
                mask(image_intensity, level, spot),
                mask(image_intensity, edge),
                mask(image_intensity, edge, ripple),
                mask(image_intensity, spot, ripple)
             ]
    
    return np.stack(masks, axis=-1)

def haze(cr, cb):
    return np.stack([mask(cr, 0, 1), mask(cb, 0, 1)],axis=-1)

def create_kernels(step=90, min=0, max=180):
    kernels = []
    
    for angle in range(min,max,step):
        rad = math.radians(angle)
        cos = math.cos(rad)
        sin = math.sin(rad)
        cos = round(cos*abs(cos),2)
        sin = round(sin*abs(sin),2)
        kernels.append(np.array([
                                    [sin-cos, 2*sin, sin+cos],
                                    [-2*cos,        0,   2*cos],
                                    [-sin-cos, -2*sin, cos-sin] 
                                ]))
        
    return kernels

#[OpenCV canny edge detection](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/canny.cpp) is predetermined to combine vertical and horizontal edge detection with no means to input custom kernels. So use our own implementation, 
# [based on a tutorial by towards datascience](https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123).

# def texture_gradients(image, *args, **kwargs):
#     kernels = create_kernels(*args, **kwargs)
#     gradients = []
    
#     for kernel in kernels:
#         gradients.append(cv2.filter2D(image, -1, kernel))
        
#     return gradients

def texture_gradients(image, d_angle):
    gradX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0) # reveal vertical edges
    gradY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1) # reveal horizontal edges

    grad = cv2.convertScaleAbs(np.sqrt(gradX**2 + gradY**2))
    angles = np.mod(np.arctan(gradY/(gradX+1e-10)) * 180/np.pi + 180, 180)  # mod() for unsigned gradients 

    axes = np.digitize(angles, np.arange(0,180,d_angle))
    
    # return np.repeat(grad.reshape(*grad.shape,1),6,axis=-1)[axes-1]

    return np.identity((180//d_angle))[axes-1]*grad.reshape(*grad.shape,1)

# For if we implement thresholds:
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def calc_thresholds(image, sigma=0.33):
    v = np.median(image)
    
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 - sigma) * v))

    return lower, upper
