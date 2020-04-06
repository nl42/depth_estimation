import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output

from Pytorch_Functions import string_of_selected_parameters

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import trange, tqdm

def simple_conv_layer(input, output, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(input, output, kernel_size=kernel_size, 
                  stride=stride, padding=padding),
        nn.ReLU()
    )

def simple_conv_block(input, output, *args, **kwargs):
    return nn.Sequential(
        simple_conv_layer(input, output, *args, **kwargs),
        simple_conv_layer(output, output, *args, **kwargs),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class SimpleCNN(nn.Module):
    def __init__(self, sizes=[3,16,32,4,10], loss_method = nn.CrossEntropyLoss(),
                 epochs=9, learning_rate=3e-3, weight_decay=1e-3, optimiser=optim.Adam):
        super().__init__()        
        self.conv_layers = nn.Sequential(
                              *[simple_conv_block(input, output)
                              for input, output in zip(sizes[:-2], sizes[1:-2])]
                            )
        
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((sizes[-2], sizes[-2])),
            nn.Flatten(),
            nn.Linear(sizes[-3]*sizes[-2]*sizes[-2], sizes[-1])
        )
        
        self.cuda()
        
        self.loss_method=loss_method
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimiser=optimiser(self.parameters(), 
                                 lr=learning_rate, weight_decay=weight_decay)
        self.train_loss_history = []
        self.val_loss_history = []
    
    def forward(self, data, train=True):
        losses = []
        accuracies = []
        
        for image, label in tqdm(iter(data), leave=False):
            image, label = image.cuda(), label.cuda()
            predicted = self.flatten(self.conv_layers(image))
            loss = self.loss_method(predicted, label)
            
            if train:
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
            
            losses.append(loss.detach().cpu().numpy())
            
            accuracies.append((predicted.max(dim=1)[1] == label).float().mean().cpu().numpy())
        
        return np.mean(losses), np.mean(accuracies)
    
    def fit(self, train_data, val_data):
        for epoch in trange(self.epochs):
            train_loss, _ = self.forward(train_data, train=True)
            self.train_loss_history.append(train_loss)

            # disable gradient calculations for validation     
            for p in self.parameters(): p.requires_grad = False

            val_loss, self.accuracy = self.forward(val_data, train=False)
            self.val_loss_history.append(val_loss)

            # enable gradient calculations for next epoch 
            for p in self.parameters(): p.requires_grad = True

            print(f"training loss: {train_loss:0.4f}"
                      f"\tvalidation loss: {val_loss:0.4f}"
                      f"\tvalidation accuracy: {self.accuracy:0.2f}")
        
        clear_output(wait=True)

    def plot_loss_history(self, *args, **kwargs):
        _,ax = plt.subplots(1,1,figsize=(20,4))
        ax.set_title(string_of_selected_parameters(sep='\n', *args, **kwargs))
        ax.plot(1+np.arange(len(self.train_loss_history)),
                self.train_loss_history)
        ax.plot(1+np.arange(len(self.val_loss_history)),
                self.val_loss_history)
        ax.grid('on')
        ax.set_xlim(left=1, right=len(self.train_loss_history))
        ax.legend(['training loss', 'validation loss']);
