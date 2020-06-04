Implementation focusing on the paper:

[3-D Depth Reconstruction from a Single Still Image](https://www.cs.cornell.edu/~asaxena/learningdepth/ijcv_monocular3dreconstruction.pdf)

# stanford:

Matlab implementation by the authors

# Implementation:

To process images first create a Laplacian object. This requires as input:

1. Initial weights
1. The function to create the feature array
1. Additional information regarding global analysis

## Weights

This is in the form of a list of arrays of shape:
1. number of scales
1. number of row combinations
1. number of values per feature

Unfortunately attempts to include scale have made this a little convoluted.

For the time being the weights should be:
1. A list one 1 array
1. The first axis of the array should have size 1
1. The second axis is the number of row combinations you want
1. The third axis is the size of that the function to create the features creates.

## Row combinations

Each row combination will be given seperate weights but each row within a row combination will have the same weights.

Each row combination contains n number of rows. If the number of row combinations is equal to the number of rows n will be 1. If the number of row combinations is 1. In the first case each row will have different weights, in the second the whole image will have the same weights.

## Function

Currently there is one function that is used, calculate local features. This should be passed as a lambda or partial as certain parameters need to be added. 

1. convert (default None)
1. squares (default False)
1. std (default True)
1. blur (default True)

### convert
The feature extraction process is carried out on a YCrCb image. This function can convert the image from it's current format to YCrCb. Input the relevant CV2 variable based on the image type. e.g. if the image is BGR use cv2.COLOR_BGR2YCrCb.

### squares
Determines whether the squared values are included in the feature vector. If not the feature length is 17 and is so 34. (useful for the third axis of the weights array).

### std

Determines whether the values are standardised.

### blur
If true a gausian blur is applied to the image before processing. As each process required the image to be gausiann blurred before being carried out on the image this is prefered. 

## Global variables

### Neighbours
A boolean that determines whether the neighbours are included.

# size_detection_ipynb

A quick attempt to try and use size as a very rough guide to depth. Thought I could get approximation size using the scale variance of Harris corner detection, didn't really work. 