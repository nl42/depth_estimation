import cv2
import numpy as np
import math
from tqdm.notebook import tqdm,trange
from itertools import product, chain, accumulate, zip_longest
from functools import partial
from operator import mul as multiply

from IPython.core.debugger import Tracer

from depth_Functions import (
    show_array_of_images,
    show_img,
    stand
)

class Iter_factory():
    def __init__(self, split, neighbours=True, patchshapes=[], columnshapes=[], patch_function=partial(np.sum,axis=(1,3)), relative_patch_function=None):
        self.split = split
        self.neighbours = neighbours
        self.patchshapes = patchshapes
        self.columnshapes = columnshapes
        self.patchfunction = patch_function
        self.relative_patch_function=relative_patch_function
    
    def absolute_iter(self,image, *args):
        return Absolute_iter(image, self.split, self.neighbours, *args, patch_function=self.patchfunction,  patchshapes=self.patchshapes, columnshapes=self.columnshapes)

    def relative_iter(self,image, *args):
        return Relative_iter(image, self.split, self.relative_patch_function, self.patchshapes, *args)
    
class Absolute_iter():
    def __init__(self, image, split, neighbours, *args, patch_function=None,  patchshapes=[], columnshapes=[]):
        self.shape = (-1,*args[0].shape)
        calc_patches = lambda image, shape : patch_function(patch_split(image, shape))
        patches = [calc_patches(image, shape) for shape in patchshapes]
        self.columns = [calc_patches(image, (math.ceil(image.shape[0]/shape[0]),shape[1])) for shape in columnshapes]
        if neighbours:
            image = calc_neighbours(image)
            patches = [calc_neighbours(patch) for patch in patches]
        self.local = np.array_split(image, split)
        self.patches = [np.array_split(scale,split) for scale in patches]
        self.args = [np.array_split(arg, split) for arg in args]
        
    def __iter__(self):
        return self
    def __next__(self):
        if len(self.local)==0:
            raise StopIteration()
        else:
            list = [self.local.pop(0)]
            if len(self.patches)>0:
                list += [patches.pop(0) for patches in self.patches]
            if len(self.columns)>0:
                list += [column for column in self.columns]
            return (np.concatenate(list, axis=None).reshape(self.shape), *[arg.pop(0).flatten() for arg in self.args])

class Relative_iter():
    def __init__(self, image, split, patch_function, patchshapes, *args):
        patches, variables = zip(*[patch_function(patch_split(image, shape), *args) for shape in patchshapes])
        self.patches = [np.array_split(scale, split) for scale in patches]
        self.variables = list(variables)
   

    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.patches[0])==0:
            self.variables.pop(0)
            self.patches.pop(0)

        if len(self.patches)==0:
            raise StopIteration()
        
        return self.patches[0].pop(0), self.variables[0] 