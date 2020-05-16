import cv2
import numpy as np
import math
from tqdm.notebook import tqdm,trange
from itertools import product, zip_longest

from IPython.core.debugger import Tracer

from depth_Functions import (
    show_array_of_images,
    show_img,
    stand,
    patch_values,
    get_neighbours,
    sum_kernel
)

class Feature_array():
    def __init__(self, image, local_function, patch_function=sum_kernel, blur=True, convert=None, dtype = np.float64, show=False, *args, **kwargs):
        if convert is None:
            self.image = image
        else :
            self.image = cv2.cvtColor(image, convert)
        if blur:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        
        self.local_function = local_function
        self.local_features = local_function(image)
        self.shape = self.local_features.shape
        self.patch_function = patch_function

    def feature_iter(self, primary, *args, **kwargs):
        iter = lambda image : [self.iterate(image, primary.shape[0], **kwargs)]
        features = iter(self.local_features)
        args = [[iter(arg)] if arg.shape[0]==self.local_features.shape[0] else arg for arg in args]
        return zip(features, primary, args)

    def iterate(self, image, size, **kwargs):
        local, patches, columns = self.calc_global(image, **kwargs)
        # Tracer()()
        local = np.array_split(local, size)
        patches = [np.array_split(subpatches, size) for subpatches in patches]
        return (zip(local, *patches), *columns)

    def calc_global(self, image, neighbours=True, patchshapes=[], columnshapes=[]):
        patches = [patch_values(image, sum_kernel(shape)) for shape in patchshapes]
        columns = [patch_values(image, sum_kernel((math.ceil(image.shape[0]/shape[0]),shape[1]))) for shape in columnshapes]
        if neighbours:
            image = get_neighbours(image)
            patches = [get_neighbours(patch) for patch in patches]
        return [image,patches, columns]

    def set_relative_hist(self, target=None, function=None, bins=11, cutoff=None, *args, **kwargs):
        if function is None:
            function = lambda patch : np.histogram(patch, bins=bins)[0]
        if target is None:
            target = self.patches
        if cutoff is None:
            cutoff = self.patches.shape[-1] // 2
        
        self.relative = np.zeros((*target.shape, bins))
        self.process_image_at_dimensions(target='relative')
        
        self.relative = np.subtract(absolute[...,1:,:], absolute[...,0:1,:])

    def show_local_features(self, channels=None, *args, **kwargs):
        if channels is None:
            channels = [i for i in range(self.local_features.shape[-1])]

        show_array_of_images([self.local_features[...,c] for c in channels], *args, **kwargs)