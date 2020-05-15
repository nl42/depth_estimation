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
    get_neighbours
)

class Feature_array():
    def __init__(self, image, local_function, patch_function=None, patchshapes=[], blur=True, convert=None, dtype = np.float64, show=False, *args, **kwargs):
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
        # if patch_function is not None:
        #     self.patchshapes = patchshapes
        #     
        #     self.patches = [patch_values(self.local_features, patch_function, shapes) for shapes in patchshapes]
        # else:
        #     self.patches = None

    def feature_iter(self, primary, *args, neighbours=True, patchshapes=[], columnshapes=[]):
        features = self.local_features
        columns = [patch_values(features, self.patch_function, ((features.shape[0]//shapes[0]),shapes[1])) for shapes in columnshapes]
        if neighbours:
            features = get_neighbours(features)
        # Tracer()()
        patches = [patch_values(features, self.patch_function, shapes) for shapes in patchshapes]
        features = np.array_split(self.local_features, primary.shape[0])
        # Tracer()()
        features = zip(features,*columns,*[np.array_split(patch, primary.shape[0]) for patch in patches])
        return zip(features, primary, *[arg if arg.shape[0]==primary.shape[0] else np.array_split(arg, primary.shape[0]) for arg in args])

    def calc_patches(self, output=None, patchshapes=None, *args, **kwargs):
        if patchshapes is None:
            patchshapes = self.patchshapes

        outputs = []
        for shape in patchshapes:
            outputs.append(self.process_patches(patchshape=shape, function=self.patch_function, *args, **kwargs))
        
        patches = np.stack(outputs, axis=-1)

        if output is None:
            self.patches = patches
        else:
            output = patches

    def process_patches(self, patchshape=None, output=None, target=None,  function=stand, name=''):
        if target is None:
            target = self.local_features
        
        if output is None:
            output = np.zeros(self.shape)

        if patchshape is None:
            heights = [0,target.shape[0]]
            widths = [0,target.shape[1]]
        else:
            heights  = [] if target.shape[0]%patchshape[0]==0 else [0]
            widths   = [] if target.shape[0]%patchshape[1]==0 else [0]
            heights += [y for y in range(0,target.shape[0],patchshape[0])]
            widths  += [x for x in range(0,target.shape[1],patchshape[1])]

        for x, (x0,x1) in tqdm(enumerate(zip(widths[:-1], widths[1:])), 
                               total=len(widths)-1, leave=False, desc=name):
                for y, (y0,y1) in enumerate(zip(heights[:-1], heights[1:])):
                    output[y0:y1,x0:x1] = function(target[y0:y1, x0:x1])
        
        return output

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

    def set_adjacent(self, dim):
        for y in range(0,self.patches.shape[0]-1):
                for x in range(0,self.patches.shape[1]-1):
                    for i, (dy,dx) in enumerate(self.neighbour_order[1:]):
                        y0 = -1-y   if dy==-1 else y
                        y1 = y      if dy== 1 else -1-y
                        x0 = -1-x   if dx==-1 else x
                        x1 = x      if dx== 1 else -1-x
                        self.patches[y0, x0, dim, i+1] = self.patches[y0+dy, x0+dx, dim, 0]
                        self.patches[y1, x1, dim, i+1] = self.patches[y1+dy, x1+dx, dim, 0]

    def __get_patchshape(self, patchshape):
        if isinstance(patchshape, int):
            patchshape = [patchshape,patchshape]
        else:
            patchshape = list(patchshape)

        while self.image.shape[0] % patchshape[0] > 0:
            patchshape[0] -= 1
            if self.image.shape[1] % patchshape[1] > 0:
                patchshape[1] -= 1
        
        while self.image.shape[1] % patchshape[1] > 0:
            patchshape[1] += 1
        
        return patchshape
    
    def get_absolute_and_relative(self, *args, **kwargs):
        return (get(['patches','columns'], *args, **kwargs), get(['relative'], *args, **kwargs))

    def show_local_features(self, channels=None, *args, **kwargs):
        if channels is None:
            channels = [i for i in range(self.local_features.shape[-1])]

        show_array_of_images([self.local_features[...,c] for c in channels], *args, **kwargs)