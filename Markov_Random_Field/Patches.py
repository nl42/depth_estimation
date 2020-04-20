import cv2
import numpy as np
import math
from scipy import ndimage
from tqdm.notebook import tqdm,trange
from itertools import product

from depth_Functions import (
    show_array_of_images,
    show_img
)

class Patches():
    def __init__(self, image, patchsize, feature_shape, convert=None, 
                 scales=[(1,1)], neighbour_order = [(0,0)], hide=False, *args, **kwargs):
        if convert is None:
            self.image = image
        else :
            self.image = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), convert)
        
        self.__get_patchsize(patchsize)
        nxm = [int(i_dim / p_dim) for i_dim, p_dim in zip(image.shape, self.patchsize)]
        self.patches = np.zeros((len(scales), *nxm, len(neighbour_order), *feature_shape))
        self.scales = scales
        self.neighbour_order  = neighbour_order
        if not hide:
            print(f'patchsize = {self.patchsize}')
            print(f'nxm = {nxm}')
            show_array_of_images([self.image], *args, **kwargs)

    def process_image_at_all_scales(self, function=np.mean, names=None):
        if names is None:
            names = [f'{dy}x{dx}' for dy,dx in self.scales]
        for scale in range(len(self.scales)):
            process_image(self, function, scale=scale, name=names[scale])

    def process_image(self, function=np.mean, scale=0, name=''):
        scale_dim = self.scales[scale]
        if scale_dim == (1,1):
            image = self.image
        else:
            image = cv2.resize(self.image, (0,0), fy=scale_dim[0], fx=scale_dim[1])
        
        heights = [y for y in range(0,image.shape[0]+1,scale_dim[0]*self.patchsize[0])]
        widths  = [x for x in range(0,image.shape[1]+1,scale_dim[1]*self.patchsize[1])]

        for y, (y0,y1) in tqdm(enumerate(zip(heights[:-1], heights[1:])), 
                               total=len(heights)-1, leave=False, desc=name):
            for x, (x0,x1) in enumerate(zip(widths[:-1], widths[1:])):
                if scale_dim == (1,1):
                    self.patches[scale, y, x] = function(image[y0:y1, x0:x1])
                else :
                    centre = (y0+int(scale_dim[0]/2)*self.patchsize[0], x0+int(scale_dim[1]/2)*self.patchsize[1])
                    for i, (y,x) in enumerate(self.neighbour_order):
                        start = (centre[0]+(y*self.patchsize[0]), centre[1]+(x*self.patchsize[1]))
                        end = (start[0]+self.patchsize[0], start[1]+self.patchsize[1])
                        self.patches[scale, y, x, i] = function(image[start[0]:end[0], start[1]:end[1]])
        
        if scale_dim == (1,1) and len(self.neighbour_order) > 1: 
            for y in range(0,math.ceil(self.patches.shape[1]/2)):
                for x in range(0,math.ceil(self.patches.shape[2]/2)):
                    for i, (dy,dx) in enumerate(self.neighbour_order[1:]):
                        self.patches[scale, y+(dy==-1), x+(dx==-1), i+1] = self.patches[scale, y+(dy==1), x+(dx==1), 0]
                        self.patches[scale, -1-(dy==1)*y, -1-(dx==1)*y, -1-i] = self.patches[scale, -1-(dy==-1)*y, -1-(dx==-1)*x, 0]
 

    
    def process_columns(self, n=4, function=lambda x: x, name=''):
        self.columns = np.zeros((n, *self.patches.shape[2:]))
        heights = [y for y in range(0,self.image.shape[0], int(self.image.shape[0]/n))]
        widths = [x for x in range(0,self.image.shape[1],self.patchsize[1])]

        for y, (y0,y1) in tqdm(enumerate(zip(heights, heights[1:])), 
                               total=len(heights)-1, leave=False, desc=name):
            for x, (x0,x1) in enumerate(zip(widths, widths[1:])):
                self.columns[y, x] = function(self.image[y0:y1, x0:x1])

    def process_patches(self, patch_number=-1, function=lambda x: x, name=''):
        for y in trange(self.patches.shape[1], leave=False, desc=name):
            for x in range(self.patches.shape[2]):
                function(self.patches[patch_number][y,x])

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

    def show_patches(self, scales=[0], neighbours=[0], split_channels=False,  *args, **kwargs):
        if split_channels:
            patches = [self.patches[s,:,:,n,c] for s,n,c in product(scales, neighbours, range(self.patches.shape[-1]))]
        else:
            patches = [self.patches[s,:,:,n] for s,n in product(scales, neighbours)]
        
        patches = [patch.repeat(self.patchsize[0],axis=0).repeat(self.patchsize[1],axis=1)
                   for patch in patches]
        show_array_of_images(patches, *args, **kwargs)
    
    # def show_patches(self, scale=0, show_neighbours=False, split_channels=True,  *args, **kwargs):
    #     
    #     # if show_neighbours:
    #     #     patches = [image for patch in patches for image in np.split(patch,patch.shape[2],axis=3)]
    #     # if split_channels:
    #     #     patches = [image for patch in patches for image in np.split(patch,patch.shape[3],axis=4)]

    #     show_array_of_images(patches.repeat(self.patchsize[0],axis=0).repeat(self.patchsize[1],axis=1), *args, **kwargs)

    def get_relative():
        self.relative = patches.copy()[:,:,1:]

        for y in trange(0,self.patches.shape[1]-1):
            for x in range(0,self.patches.shape[2]-1):
                for i, (dy,dx) in enumerate(self.neighbour_order[1:]):
                    self.relative[scale, y+(dy==-1), x+(dx==-1), i] = self.patches[scale, y+(dy==-1), x+(dx==-1), 0] - self.patches[scale, y+(dy==1), x+(dx==1), 0]