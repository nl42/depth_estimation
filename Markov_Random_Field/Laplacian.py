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

from depth_Functions import updating_mean, stand

from functools import partial

from IPython.core.debugger import Tracer

class Laplacian():
    def __init__(self, initial_weights, local_function):
        self.initial_weights = np.array(initial_weights)
        self.local_function = local_function
        self.training_count=0
        self.weights = np.ones(initial_weights.shape)

    def calc_features(self, image):
        return Feature_array(image, convert=cv2.COLOR_BGR2YCrCb, local_function=self.local_function)

    def train_combined(self, train_images, train_labels, *args, **kwargs):
        combined_train_image = np.concatenate(train_images,axis=1)
        combined_train_labels = np.concatenate(train_labels,axis=1)
        self.train([combined_train_image], [combined_train_labels], *args, **kwargs)

    def train(self, train_images, train_labels, function=None, prep=np.log, conc_function=updating_mean):
        if function is None:
            function = self.linear_function
        # if prep is not None:
        # train_labels = prep(train_labels)
        function = partial(self.__train_function, function)
        training_weights = []
        for image, labels in tqdm(zip(train_images, train_labels), total=len(train_images), leave=False):
            features_array = self.calc_features(image)
            if prep is not None:
                labels = prep(labels)
            # Tracer()()
            weights, covariance = zip(*[curve_fit(function, partial_features, partial_labels.flatten(), p0=partial_weights.flatten()) 
                                   for partial_features, partial_weights, partial_labels in tqdm(features_array.feature_iter(self.initial_weights, labels),
                                   total=self.weights.shape[0], leave=False, desc='local weights')])
            # self.weights = np.array(weights)
            conc_function(self.weights, weights, self.training_count)
            # training_weights.append(np.array(weights))
            self.training_count += 1
    
        # self.weights = np.mean(np.stack(training_weights),axis=0)

    def post(image, minmax):
        return cv2.normalize(image, None, *minmax, cv2.NORM_MINMAX)

    def predict(self, image, function=None, post=None, *args):
        if function is None:
            function = self.exponential_function
        
        feature_array = self.calc_features(image) 
        prediction = np.concatenate([function(features, weights) for features, weights in feature_array.feature_iter(self.weights)])
        
        prediction[prediction>1] = 1

        if post is None:
            return prediction
        else:
            return post(prediction, *args)

    def exponential_function(self, features, weights):
        return np.exp(features @ weights)

    def linear_function(self, features, weights):
        # Tracer()()
        return (features @ weights)
    
    def __train_function(self, function, x, *params):
        params = np.array(params)
        return function(x, params).flatten()
