import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
from inspect import getsource
from sklearn.preprocessing import StandardScaler
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

    if (image_array_shape[0]==1 or image_array_shape[1] == 1):
        for index, image in enumerate(images):
            show_img(image, title=titles[index], *args, **kwargs)
    else:
        _,axis = plt.subplots(*image_array_shape, figsize=figsize[::-1])
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
    image[image == 1] = 2
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

# def stand(array):
#     if isinstance(array, list):
#         array = np.array(array)

#     mean, std = np.mean(array, axis=(0,1)), np.std(array, axis=(0,1))

#     std = [1 if s==0 else s for s in std]

#     return (array - mean) / std
