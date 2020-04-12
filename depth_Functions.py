import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def show_img(image, title='', axis=None, heatmap=False, depthmap=False, figsize=(8,8)):
    if not axis: _,axis = plt.subplots(1,1,figsize=figsize[::-1])
        
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    
    if len(image.shape)==3: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif heatmap:
        image = cv2.applyColorMap(255-image, cv2.COLORMAP_JET)
    elif depthmap:
        image = cv2.applyColorMap(255-image, cv2.COLORMAP_JET)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    axis.imshow(image)
    axis.set_title(title)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    return axis

def show_array_of_images(images, shape=None, figsize=(16,16), 
                         titles=[], *args, **kwargs):
    if hasattr(images, 'shape'):
        if shape is None:
            shape = images.shape[0:2] 
        images = images.flatten()
    elif shape is None:
        if len(images) > 4:
            sqrt = math.sqrt(len(images))
            shape = (math.ceil(sqrt), math.floor(sqrt))
        else :
            shape = (len(images),1)
    
    for i in range(len(images)-len(titles)):
        titles.append('')

    if (shape[0]==1 or shape[1] == 1):
        for index, image in enumerate(images):
            show_img(image, title=titles[index], *args, **kwargs)
    else:
        _,axis = plt.subplots(*shape, figsize=figsize[::-1])
        for index, image in enumerate(images):
            show_img(image, axis=axis[int(index/shape[1])][int(index%shape[1])], 
                     title=titles[index], *args, **kwargs)

def import_raw_colour_image(path):
    with open(path, 'rb') as f:
        buffer = f.read()
    image = np.frombuffer(buffer, dtype="uint8").reshape(int(720),1280,-1)
    return image[...,:-1][...,::-1]

def import_raw_depth_image(path):
    with open(path, "rb") as f:
        buffer = f.read()
        
    image = np.frombuffer(buffer, dtype="float32").reshape(int(720), -1)
    return image
