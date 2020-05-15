import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.interpolation import shift
from inspect import getsource
from IPython.display import Code

def show_img(image, title='', axis=None, heatmap=False, depthmap=False, figsize=(8,16)):
    if axis is None:
        _,axis = plt.subplots(1,1,figsize=figsize[::-1])

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, 32).astype(np.uint8)
        
    if image.shape[-1]==3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif heatmap:
        image = cv2.applyColorMap(255-image, cv2.COLORMAP_JET)
    elif depthmap:
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    axis.imshow(image)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    return axis

def show_array_of_images(images, split_channels=False, image_array_shape=None, 
                         figsize=(16,32), titles=[], *args, **kwargs):
    if hasattr(images, 'shape'):
        print('for now only show lists of images')
        return
        # if split_channels:
        #     np.split(images, axis=-1)
        # if image_array_shape is None:
        #     image_array_shape = images.shape[0:2] 
        # images = images.flatten()
    else:
        if split_channels:
            images = [channel for image in images for channel in np.split(image,image.shape[-1], axis=-1) ]
        
        if image_array_shape is None:
            if len(images) > 2:
                sqrt = math.ceil(math.sqrt(len(images)))
                image_array_shape = (sqrt, math.ceil(len(images)/sqrt))
            else :
                image_array_shape = (len(images),1)
    
    for i in range(len(images)-len(titles)):
        titles.append('')

    _,axis = plt.subplots(*image_array_shape, figsize=figsize[::-1])

    if (image_array_shape[0] == 1 or image_array_shape[1] == 1):
        for index, image in enumerate(images):
            show_img(image, axis=axis[index], title=titles[index], *args, **kwargs)
    else:
        for index, image in enumerate(images):
            show_img(image, axis=axis[int(index/image_array_shape[1])][int(index%image_array_shape[1])], 
                     title=titles[index], *args, **kwargs)

def import_raw_colour_image(path):
    with open(path, 'rb') as f:
        buffer = f.read()
    image = np.frombuffer(buffer, dtype="uint8").reshape(int(720),1280,-1)
    return image[...,:-1][...,::-1]

def import_raw_depth_image(path):
    with open(path, "rb") as f:
        buffer = f.read()
        
    image = 1-np.frombuffer(buffer, dtype="float32").reshape(int(720), -1)
    return image

def show_function(function):
    
    return Code(getsource(function), language='python')

def stand(array):
    if isinstance(array, list):
        array = np.array(array)
    
    shape = array.shape
    array = array.reshape((-1, shape[-1]))

    scalar = StandardScaler()
    return scalar.fit_transform(array).reshape(shape)

def updating_mean(existing, new, count):
    existing[:] = ((existing * count) + new) / (count+1)

def sum_patch(patchshape):
        return np.ones(patchshape)
    
def patch_values(image, function, patchshape, stride=None):
    if type(patchshape) == int:
        patchshape = (patchshape, patchshape)
    if stride is None:
        stride = patchshape
    elif type(stride) == int:
        stride = (stride,stride)

    kernel = function(patchshape)

    start = (image.shape[0]%stride[0], image.shape[1]%stride[1])

    heights  = [] if start[0]==0 else [patchshape[0]//2]
    widths   = [] if start[1]==0 else [patchshape[0]//2]
    heights += [y for y in range(start[0]+patchshape[0]//2,image.shape[0],stride[0])]
    widths  += [x for x in range(start[1]+patchshape[1]//2,image.shape[1],stride[1])]

    return cv2.filter2D(image, -1, kernel)[heights][:,widths]

def get_neighbours(image, n=1):
    [up, right, down, left] = [image.copy() for i in range(4)]
    up[n:]          = image[:-n]
    right[:,:-n]    = image[:,n:]
    down[:-n]       = image[n:]
    left[:,n:]      = image[:,:-n]    
    return np.stack([image, up, right, down, left], axis=2)

def calculate_relative(image, n=1):
    [up, right, down, left] = [np.zeros(image.shape) for i in range(4)]
    up[n:] = np.diff(image, n=n, axis=0)
    right[:,:-n] = np.diff(image[::-1], n=n, axis=1)[::-1]
    down[:-n] = np.diff(image[::-1], n=n, axis=0)[::-1]
    left[:,n:] = np.diff(image, n=n, axis=1)
    return np.stack([up,right,down,left], axis=0)

# def stand(array):
#     if isinstance(array, list):
#         array = np.array(array)

#     mean, std = np.mean(array, axis=(0,1)), np.std(array, axis=(0,1))

#     std = [1 if s==0 else s for s in std]

#     return (array - mean) / std
