import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import tqdm,trange
from itertools import product, zip_longest

from depth_Functions import (
    show_array_of_images,
    show_img
)

class Patches():
    type_patch = 0
    type_column = 1
    types = {type_patch : 'patch', type_column : 'column'}

    def __init__(self, image, patchsize, feature_shape, convert=None, function=np.mean,
                 dimensions=[(0,1,1),(1,1,1),(0,3,3), (0,9,9)], 
                 neighbour_order = [(0,0), (-1,0), (0,1), (1,0), (0,-1)],
                 dtype = np.float64, hide=False, *args, **kwargs):
        if convert is None:
            self.image = image
        else :
            self.image = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), convert)
        
        self.__get_patchsize(patchsize)
        nxm = [i_dim // p_dim for i_dim, p_dim in zip(image.shape, self.patchsize)]
        self.patches = np.zeros((*nxm, len(dimensions), len(neighbour_order), *feature_shape), dtype=dtype)
        self.dimensions = dimensions
        self.neighbour_order  = neighbour_order
        self.function = function
        if not hide:
            print(f'patchsize = {self.patchsize}')
            print(f'dimensions = {[(self.types[type_int],y,x) for type_int,y,x in dimensions]}')
            print(f'shape = {self.patches.shape}')
            show_array_of_images([self.image], *args, **kwargs)

    def process_image_at_dimensions(self, names=None, dims=None, *args, **kwargs):
        if dims is None:
            dims = range(len(self.dimensions))
        if names is None:
            names = [f'{type}: {dy}x{dx}' for type,dy,dx in self.dimensions]

        for dim in dims:
            self.process_image(dim=dim, name=names[dim], *args, **kwargs)

    def process_image(self, function=None, patchsize=None, target=None, dim=0, name=''):
        if patchsize is None:
            patchsize = self.patchsize
        if target is None:
            target = self.patches
        if function is None:
            function = self.function
        
        dim_type, y_scale, x_scale = self.dimensions[dim]

        if y_scale == 1 == x_scale :
            image = self.image
        else:
            image = cv2.resize(self.image, (0,0), fy=y_scale, fx=x_scale)
        
        if dim_type == self.type_column:
            y_step = image.shape[0]//len(self.neighbour_order)
        else:
            y_step = y_scale*patchsize[0]

        heights = [y for y in range(0,image.shape[0],y_step)              ]+[image.shape[0]]
        widths  = [x for x in range(0,image.shape[1],x_scale*patchsize[1])]+[image.shape[1]]

        for x, (x0,x1) in tqdm(enumerate(zip(widths[:-1], widths[1:])), 
                               total=len(widths)-1, leave=False, desc=name):
                for y, (y0,y1) in enumerate(zip(heights[:-1], heights[1:])):
                    if dim_type == self.type_column:
                        target[:, x, dim, y] = function(image[y0:y1, x0:x1])
                    elif y_scale == 1 == x_scale:
                        target[y, x, dim, :] = function(image[y0:y1, x0:x1])
                    else :
                        centre = (y0+(y_scale//2)*patchsize[0], x0+(x_scale//2)*patchsize[1])

                        for i, (dy,dx) in enumerate(self.neighbour_order):
                            start = (centre[0]+(dy*patchsize[0]), centre[1]+(dx*patchsize[1]))
                            end = (start[0]+patchsize[0], start[1]+patchsize[1])
                            target[y, x, dim, i] = function(image[start[0]:end[0], start[1]:end[1]])
        
                if y_scale == 1 == x_scale and len(self.neighbour_order) > 1: 
                    self.set_adjacent(dim=0)

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

    def __get_patchsize(self, patchsize):
        if isinstance(patchsize, int):
            patchsize = [patchsize,patchsize]
        else:
            patchsize = list(patchsize)

        while self.image.shape[0] % patchsize[0] > 0:
            patchsize[0] -= 1
            if self.image.shape[1] % patchsize[1] > 0:
                patchsize[1] -= 1
        
        while self.image.shape[1] % patchsize[1] > 0:
            patchsize[1] += 1
        
        self.patchsize = patchsize
        

    def extend(shape):
        for (axis, repeats) in enumerate(shape):
            if repeats > 1:
                self.patches.repeat(repeats, axis=2+axis)

    def get(self, attribute_names, shape, axis=0):
        
        attributes = [getattr(self, name).reshape(shape) for name in attribute_names]

        return np.concatenate(attributes, axis=0)
    
    def get_absolute_and_relative(self, *args, **kwargs):
        return (get(['patches','columns'], *args, **kwargs), get(['relative'], *args, **kwargs))

    def show(self, target='patches', bordersize=0, subborder=(0,0), dims=[0], neighbours=[0], channels=None, *args, **kwargs):
        patches = getattr(self, target)
        patchsize = (self.image.shape[0]//patches.shape[0], self.image.shape[1]//patches.shape[1])
        output = patches.repeat(patchsize[0], 0).repeat(patchsize[1], 1)
        
        if bordersize != 0:
            for y in range(0,output.shape[0]+1,patchsize[0]):
                output[y-bordersize:y,:] = 0
                output[y:y+bordersize,:] = 0
            
            for x in range(0,output.shape[1]+1,patchsize[1]):
                output[:,x-bordersize:x] = 0
                output[:,x:x+bordersize] = 0
        
        if subborder != (0,0):
            for y in range(0,output.shape[0]+1,subborder[0]*patchsize[0]):
                output[y-bordersize:y+bordersize,:] = 255
            
            for x in range(0,output.shape[1]+1,subborder[1]*patchsize[1]):
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
        elif len(image.shape) == 3:
            outputs = [output[:,:,c] for c in channels]

        show_array_of_images(outputs, *args, **kwargs)