import numpy as np
import math
from numpy import errstate,isneginf,array
import cv2

from scipy.optimize import curve_fit 
from IPython.display import clear_output
from tqdm.notebook import tqdm, trange

from functools import partial

from Global import calc_features, unravel_patches

from Feature_Extraction import calculate_local_features
from depth_Functions import updating_mean, iterate

from IPython.core.debugger import Tracer

class Laplacian():
    def __init__(self, initial_weights, local_function, neighbours=True, patch_function=np.sum, relative_scales=None):
        self.initial_weights = np.array(initial_weights)
        self.local_function = local_function
        self.training_count=0
        self.weights = np.ones(initial_weights.shape)
        self.global_args = (neighbours, patch_function, relative_scales)

    def train(self, train_images, train_labels, train_function=None, prep=np.log, conc_function=updating_mean, *global_args):
        if train_function is None:
            train_function = self.linear_function
        if len(global_args) == 0:
            global_args = self.global_args
        
        train_function = partial(self.__train_function, train_function)
        
        for image, labels in tqdm(zip(train_images, train_labels), total=len(train_images), leave=False):
            if prep is not None:
                labels = prep(labels)
            features = calc_features(image, self.local_function, *global_args)
            # Tracer()()
            weights, covariance = zip(*[curve_fit(train_function, xdata=partial_features, ydata=partial_labels.flatten(), p0=partial_weights.flatten(), method='trf') 
                                        for partial_weights, partial_features,  partial_labels in 
                                        tqdm(iterate(self.initial_weights, features, labels), total=self.initial_weights.shape[0], leave=False, desc='training')])
            
            conc_function(self.weights, weights, self.training_count)
            self.training_count += 1

    def predict(self, image, post=None, neighbours=True, function=None, patch_function=np.sum, *global_args):
        if function is None:
            function = self.exponential_function
        if len(global_args) == 0:
            global_args = self.global_args

        features = calc_features(image, self.local_function, *global_args)
        prediction = [function(partial_features, partial_weights.flatten()) 
                      for partial_weights, partial_features   in tqdm(iterate(self.weights, features),
                      total=self.weights.shape[0], leave=False, desc='prediction')]
        # prediction[prediction>1] = 1
        if post is None:
            return np.concatenate(prediction).reshape(image.shape[0:2])
        return post(prediction)

    # def linear_function(self, inputs):
    #     return np.sum([f @ w for f,w in inputs], axis=0)

    def exponential_function(self, features, weights):
        return np.exp(self.linear_function(features, weights))
    
    def linear_function(self, features, weights):
        features = features.reshape(*features.shape[:2], -1)
        # weights = np.array(weights).reshape(features.shape[-1], -1)
        # Tracer()()
        return features @ weights
    
    def __train_function(self, function, x, *params):
        # params = np.array(params).reshape(x.shape[-1])
        return function(x, params).flatten()
