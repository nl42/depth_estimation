import numpy as np
import math
from numpy import errstate,isneginf,array
import cv2

from scipy.optimize import curve_fit 
from IPython.display import clear_output
from tqdm.notebook import tqdm, trange

from functools import partial

from Global import calc_features, calc_scales, unravel_patches

from Feature_Extraction import calculate_local_features
from depth_Functions import updating_mean, iterate

from IPython.core.debugger import Tracer

def least_squares(function, *args):
    return np.concatenate([function(*subargs, primary) for primary, *subargs in tqdm(iterate(*args), total=len(args[0]), leave=False, desc='training')],axis=0)

def scaled_least_squares(function, *args):
    return [least_squares(function, *scaledargs) for scaledargs in tqdm(zip(*args), total=len(args[0]), leave=False, desc='scales')]
            

class Laplacian():
    def __init__(self, initial_weights, local_function, neighbours=True, patch_function=np.sum, initial_combined_weights=None, relative_scales=[(1,1)]):
        self.initial_weights = [np.array(weights) for weights in initial_weights]
        self.initial_combined_weights = initial_combined_weights
        self.local_function = local_function
        self.training_count=0
        self.neighbours = neighbours
        self.global_args = (patch_function, relative_scales)

    def train(self, train_images, train_labels, train_function=None, predict_function=None, prep=np.log, conc_function=updating_mean, *global_args):
        if train_function is None:
            train_function = self.linear_function
        if len(global_args) == 0:
            global_args = self.global_args
        
        least_squares_function = lambda x,y,params : np.array(curve_fit(partial(self.__train_function, train_function),x.reshape(-1, len(params)),y.flatten(),params)[0]).reshape(1,-1)

        for image, labels in tqdm(zip(train_images, train_labels), total=len(train_images), leave=False):
            if prep is None:
                prept_labels = labels
            else:
                prept_labels = prep(labels)
            features = self.local_function(image)
            global_features  = calc_scales(features, self.neighbours, *global_args)
            global_labels = calc_scales(prept_labels, False, *global_args)
            # Tracer()()
            
            weights = [np.array(scaled_least_squares(least_squares_function, weights, global_features, global_labels)) for weights in self.initial_weights]
            # Tracer()()
            if self.training_count == 0:
                self.weights = weights
            else:
                # Tracer()()
                self.weights = [conc_function(existing, new, self.training_count) for existing, new in zip(self.weights, weights)]

            if self.initial_combined_weights is not None:
                # Tracer()()
                predicted_patches = [scaled_least_squares(train_function, w, global_features) for w in weights]
                predicted_images = np.stack([unravel_patches(p, image.shape[0:2]) for pred in predicted_patches for p in pred], axis=-1)
                # predicted_images[predicted_images>1]=1
                # Tracer()()
                combined_weights = least_squares(least_squares_function, self.initial_combined_weights, predicted_images, labels)

                if self.training_count == 0:
                    self.combined_weights = combined_weights
                else:
                    conc_function(self.combined_weights, combined_weights, self.training_count)
            # Tracer()()
            
            self.training_count += 1

    def predict(self, image, post=None, neighbours=True, function=None, patch_function=np.sum, *global_args):
        if function is None:
            function = self.exponential_function
        if len(global_args) == 0:
            global_args = self.global_args

        features = self.local_function(image)
        global_features = calc_scales(features, self.neighbours, *global_args)
        # Tracer()()
        predictions = [scaled_least_squares(function, weights, global_features) for weights in self.weights]
        predictions = [unravel_patches(p, image.shape[0:2]) for pred in predictions for p in pred]

        if len(predictions)==1: 
            final_prediction = predictions[0]
        else:
            # final_prediction = np.stack(predictions,axis=-1)
            final_prediction = least_squares(self.linear_function, self.combined_weights, np.stack(predictions,axis=-1))

        if post is None:
            return final_prediction

        return post(final_prediction)

    # def linear_function(self, inputs):
    #     return np.sum([f @ w for f,w in inputs], axis=0)

    def exponential_function(self, features, weights):
        return np.exp(self.linear_function(features, weights))
    
    def linear_function(self, features, weights):
        # features = features.reshape(-1,len(weights))
        # weights = np.array(weights).reshape(features.shape[-1], -1)
        # Tracer()()
        return features @ weights
    
    def __train_function(self, function, x, *params):
        # params = np.array(params).reshape(x.shape[-1])
        return function(x, params).flatten()
