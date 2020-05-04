import cv2
import numpy as np
import math
from tqdm.notebook import tqdm,trange
from itertools import product, zip_longest

from depth_Functions import (
    show_array_of_images,
    show_img,
    stand
)

class Patches():
    def __init__(self, image, local_function, global_function=None, patchshapes=[], convert=None, dtype = np.float64, *args, **kwargs):
        if convert is None:
            self.image = image
        else :
            self.image = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), convert)
        
        self.local_function = local_function
        self.local_features = local_function(image)
        self.shape = self.local_features.shape
        self.patchshapes = patchshapes
        self.global_function = global_function
        self.global_features = None
    
    def get_features(self):
        if self.global_function is None:
            return self.local_features
        if self.global_features is None:
            self.calc_global_features()
        
        return np.append(self.local_features, self.global_features.reshape((*self.local_features.shape[0:-1],-1)), axis=-1)[...,None]

    def calc_global_features(self, output=None, patchshapes=None, *args, **kwargs):
        if patchshapes is None:
            patchshapes = self.patchshapes

        outputs = []
        for shape in patchshapes:
            outputs.append(self.process_patches(patchshape=shape, function=self.global_function, *args, **kwargs))
        
        global_features = np.stack(outputs, axis=-1)

        if output is None:
            self.global_features = global_features
        else:
            output = global_features

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
        

    def extend(shape):
        for (axis, repeats) in enumerate(shape):
            if repeats > 1:
                self.patches.repeat(repeats, axis=2+axis)

    def get(self, attribute_names, shape, axis=0):
        
        attributes = [getattr(self, name).reshape(shape) for name in attribute_names]

        return np.concatenate(attributes, axis=0)
    
    def get_absolute_and_relative(self, *args, **kwargs):
        return (get(['patches','columns'], *args, **kwargs), get(['relative'], *args, **kwargs))

    def show(self, target='features', bordersize=0, subborder=(0,0), dims=[0], neighbours=[0], channels=None, *args, **kwargs):
        patches = getattr(self, target)
        patchshape = (self.image.shape[0]//patches.shape[0], self.image.shape[1]//patches.shape[1])
        output = patches.repeat(patchshape[0], 0).repeat(patchshape[1], 1)
        
        if bordersize != 0:
            for y in range(0,output.shape[0]+1,patchshape[0]):
                output[y-bordersize:y,:] = 0
                output[y:y+bordersize,:] = 0
            
            for x in range(0,output.shape[1]+1,patchshape[1]):
                output[:,x-bordersize:x] = 0
                output[:,x:x+bordersize] = 0
        
        if subborder != (0,0):
            for y in range(0,output.shape[0]+1,subborder[0]*patchshape[0]):
                output[y-bordersize:y+bordersize,:] = 255
            
            for x in range(0,output.shape[1]+1,subborder[1]*patchshape[1]):
                output[:,x-bordersize:x+bordersize] = 255

        # If you want an image for every instance of an axis, set it to None
        if channels is None:
            channels = range(output.shape[-1])
        if dims is None:
            dims = range(output.shape[2])
        if neighbours is None:
            neighbours = range(output.shape[3])

        if len(output.shape) == 5:
            outputs = [output[:,:,d,n,c] for d,n,c in product(dims, neighbours, channels)]
        elif len(output.shape) == 3:
            outputs = [output[:,:,c] for c in channels]

        show_array_of_images(outputs, *args, **kwargs)