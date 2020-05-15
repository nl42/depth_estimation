from scipy.optimize import curve_fit 

from Feature_Extraction import create_local_feature_vector

from Feature_array import Feature_array

from IPython.display import clear_output

from tqdm.notebook import tqdm, trange

from scipy.odr import ODR, Data, Model

import numpy as np
import math
from numpy import errstate,isneginf,array

import cv2

from IPython.core.debugger import Tracer

class Laplacian():
    def __init__(self, initial_weights, initial_var_weights, local_function, z=1, global_function=None, patchshapes=None):
        self.z = z
        self.weights = np.array(initial_weights)
        # self.var_weights = np.array(initial_var_weights)
        # self.relative_weights = np.stack([initial_weights for i in range(4)],axis=0)
        # self.relative_var_weights = np.stack([initial_var_weights for i in range(4)],axis=0)
        self.local_function = local_function
        self.global_function = global_function
        self.patchshapes = patchshapes

    def create_patch(self, image):
        return Feature_array(image, convert=cv2.COLOR_BGR2YCrCb, local_function=self.local_function, global_function=self.global_function, patchshapes=self.patchshapes)

    def train(self, train_images, train_labels):
        for image, labels in tqdm(zip(train_images, train_labels), total=len(train_images), leave=False):
            patch = self.create_patch(image)
            features = patch.get_features()
            variance = self.__calc(self.__least_squares, -features, self.weights, np.log(labels))
            # self.__calc(self.__least_squares, features, self.var_weights, variance, bounds=[0,np.inf], feature_function=lambda x : np.var(x, axis=(0,1)))
            # relative_features = calculate_relative(features)
            # relative_labels = calculate_relative(labels)
            # # Tracer()()
            # # relative_variances = [self.__calc(self.__least_squares, features, weights, labels, method=self.__exponential_function)
            # #                       for relative_features, weights, labels in tqdm(zip(calculate_relative(features), self.relative_weights, relative_labels),
            # #                       total = 4, leave=False, desc='relative weights')]
            
            # [self.__calc(self.__least_squares, features, weights, labels, bounds=[0,np.inf], feature_function=lambda x : np.var(x, axis=(0,1)), method=self.__exponential_function,)
            #  for relative_features, weights, labels in tqdm(zip(calculate_relative(features), self.relative_var_weights, relative_variances),
            #  total = 4, leave=False, desc='relative weights')]
    
    def train_exp_linear(self, train_images, train_labels):
        for image, labels in tqdm(zip(train_images, train_labels), total=len(train_images), leave=False):
            patch = self.create_patch(image)
            features = patch.get_features()
            self.__calc(self.__least_squares, features, self.weights, np.log(labels), method=self.__linear_and_log)


    def predict(self, image, local=True, relative=True):
        features = self.create_patch(image).get_features()
        
        if not relative:
            return self.__calc(self.__local, features, self.weights)
        
        if not local:
            return [self.__calc(self.__local, features, weights)
                    for features, weights in zip(calculate_relative(features), self.relative_weights)]

        e1 = self.__calc(self.__variance_weighting, features, self.weights, self.var_weights)
        e2 = [self.__calc(self.__variance_weighting, features, weights, var_weights)
              for features, weights, var_weights in zip(calculate_relative(features), self.relative_weights, self.relative_var_weights)]
        # Tracer()()
        return np.exp(e1 + np.mean(e2, axis=0))
    
    def predict_linear_and_exp(self, image):
        features = self.create_patch(image).get_features()
        return self.__calc(self.__combined, features, self.weights)

    def __least_squares(self, features, weights, labels, bounds=(-np.inf,np.inf), feature_function=None, method=None):       
        if feature_function is not None:
            # Tracer()()
            features = feature_function(features)
        # features = features.reshape(-1,features.shape[-1])
        if method is None:
            method = self.__linear_function
        params = curve_fit(method, features, labels.flatten(), p0=weights.flatten(), method='trf', bounds=bounds, max_nfev=10**6)
        weights[:] = params[0].reshape(weights.shape)
        covariance = params[1]
        return np.diag(covariance)

    # def __full_exponential_function(self, features, weights, var_weights, relative_weights, relative_var_weights):
    #     e1 = -features @ (weights * (np.var(features, axis=(0,1)) @ var_weights))
    #     relative_features = relative(features)
    #     e2 = -relative_features @ (relative_weights * (np.var(features, axis=(0,1)) @ relative_var_weights)) 
    #     return np.exp(e1 - e2)

    def __variance_weighting(self, features, weights, var_weights):
        return -1*features @ (weights * (np.var(features, axis=(0,1)) @ var_weights))

    def __local(self, features, weights):
        return np.exp(-features @ weights)

    def __exponential_function(self, features, *weights):
        weights = np.array(weights).reshape(features.shape[-1],-1)
        return np.exp(features @ weights).flatten()

    def __linear_function(self, features, *weights):
        weights = np.array(weights).reshape(features.shape[-1],-1)
        return (features @ weights).flatten()
    
    def __linear_and_log(self, features, *weights):
        linear = np.array(weights[:len(weights)//2]).reshape(features.shape[-1],-1)
        exp = np.array(weights[len(weights)//2:]).reshape(features.shape[-1],-1)
        return ((features @ linear) + np.log(-features @ exp)).flatten()
    
    def __combined(self, features, linear, exp):
        return ((features @ linear) + np.exp(-features @ exp)).flatten()

    # def __transposed_linear_function(self, features, *weights):
    #     # features = features.transpose()
    #     return (weights @ features).flatten()
    
    def __calc(self, function, features, weights, *args, **kwargs):
        # Tracer()()
        dy = features.shape[0] // weights.shape[0]
        dx = features.shape[1] // weights.shape[1]
        
        yr = features.shape[0] % weights.shape[0]
        xr = features.shape[1] % weights.shape[1]
        
        row_numbers = [y*(dy+1) for y in range(0, yr+1)]
        row_numbers += [y for y in range(row_numbers[-1]+dy, features.shape[0]+1,dy)]
        
        col_numbers = [x*(dx+1) for x in range(0, xr+1)]
        col_numbers += [x for x in range(col_numbers[-1]+dx, features.shape[1]+1,dx)] 

        output = []

        for y,(y0,y1) in tqdm(enumerate(zip(row_numbers[:-1], row_numbers[1:])), total=len(row_numbers)-1, leave=False):
            output.append([])
            for x,(x0,x1) in enumerate(zip(col_numbers[:-1], col_numbers[1:])):
                output[-1].append(function(features[y0:y1, x0:x1], weights[y,x], 
                                           *[arg[y0:y1, x0:x1] if arg.shape[0:2] == features.shape[0:2] else arg[y,x] for arg in args], 
                                           **kwargs))
        if len(output[0][0].shape)==1:                                   
            return np.array(output)
        else:
            return np.block(output)
        
    
def calculate_relative(features, n=1):
    [up, right, down, left] = [np.zeros(features.shape) for i in range(4)]
    up[n:] = np.diff(features, n=n, axis=0)
    right[:,:-n] = np.diff(features[::-1], n=n, axis=1)[::-1]
    down[:-n] = np.diff(features[::-1], n=n, axis=0)[::-1]
    left[:,n:] = np.diff(features, n=n, axis=1)
    return np.stack([up,right,down,left], axis=0)