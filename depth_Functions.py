import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

feature_descriptors =   {
                                'SIFT'  :   cv2.xfeatures2d.SIFT_create(), 
                                'SURF'  :   cv2.xfeatures2d.SURF_create(upright=False, extended=True), 
                                'ORB'   :   cv2.ORB_create(nfeatures=10000), 
                                'BRISK' :   cv2.BRISK_create()
                        }

def show_img(im, ax=None, figsize=(8,8)):
    if not ax: _,ax = plt.subplots(1,1,figsize=figsize)
    # if len(im.shape)==2: im = np.tile(im[:,:,None], 3)
    if len(im.shape)==3: im = im[:,:,::-1]
    ax.imshow(im)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    return ax


def show_list_of_images(images, height, width, figsize=(16,16)):
    _,ax = plt.subplots(height, width, figsize=figsize)

    for index, image in enumerate(images):
        if image.dtype == 'float64':
            image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
        show_img(image, ax[int(index/width)][index%width])

def show_array_of_images(images, figsize=(16,16)):
    _,ax = plt.subplots(images.shape[0], images.shape[1], figsize=figsize)
    
    for row_number, row in enumerate(array):
        for column_number, image in enumerate(row):
            show_img(image, ax=ax[row_number][column_number])

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

def plot_image_descriptors(image, key_points):
    image_copy = image.copy()
    cv2.drawKeypoints(image, key_points, image_copy, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_img(image_copy)
