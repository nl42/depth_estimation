By importing SimpleCNN you can create a class that represents your model

**N.B. Due to the way pickle works: to save the model you appear to have to use 'Import SimpleCNN' and call SimpleCNN.SimpleCNN() instead of using 'from SimpleCNN import SimpleCN**

SimpleCNN.ipynb was a notebook I was using to experiment with parameters, may help with understanding how to implement.

# SimpleCNN - (inherits from nn.Module)

SimpleCNN is designed to a model using a set convblocks (2 conv layers with Relu activation followed by MaxPool) finished with an AdaptivePool layer and then flattening the data:

## init - Creates model:

## Stored values:

1. loss method
1. epochs
1. learning_rate
1. optimiser
1. train_loss_history
1. val_loss_history
1. accuracy (once trained)
1. Inherited values from nn.Module

### inputs {defaults}:
1. sizes (list) {3,16,32,4,10}
1. loss method {Cross Entropy}
1. epochs {9}
1. learning rate {3e-3}
1. weight decay {1e-3}
1. optimiser {Adam}

### sizes:

    1. Calculates input output channels from the n-2 of sizes:
        {block1 (input 3, output 16 ) -> block2 (input 16, output 32)}

    1. Penultimate value is the dimensions set by the AdaptivePool function

    1. Final value is the outputs, which should be equal to the number of categories


## Forward:
    
Defines how the model processes data passed through to it during training/validation. Can be left alone unless causing errors

## fit:

Use to pass data through to the model to train. 

## plot_loss_history:

Use to plot the loss history of the model for training and validation

The title of the table created will show the parameter input (set to True):
    e.g. model.plot_loss_history(acc=True) -> title will show accuracy of the model

# Pytorch_Functions

## string_of_selected_parameters:

You shouldn't need to call this directly.

Has list of parameters that can be called.

## save_models:

Save a list of models to a directory. input the model to save and the directory path.

This method will use string_of_selected_parameters to create the model's file name, set parameters to True that should be included (atm don't include optimiser, which is a class)

## load_models:

Load all files with .pth extension in a given directory as models in a list. 

## get_stats:

Calculates the mean and standard deviation of a given dataloader

