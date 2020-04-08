# Notes

- ###### Depth represented on log scale
- ###### Laplasian outperforms Gaussian
- ###### Many cues were given large weights
    - Training requires sufficient cues

# Depth Features

Convert the image from RGB/BGR to YCrCb:
    1. Y: Intensity
    1. Cr: red difference
    1. Cb: blue difference

1. ### texture variation [9 channels]
    - Apply 9 Law's masks to Y    
1. ### texture gradient [6 channels]:
    - Apply 6 edge filters to Y
1. ### haze [2 channels]:
    - Apply 1st Law's Mask to Cr,Cb
1. ### fog (not implemented)
    - light scattering

# Absolute Depth

1. ### Scale variance
1. ### Neighbour comparison
1. ### Column comparison

# [Markov Random Field](https://ermongroup.github.io/cs228-notes/representation/undirected/)

##### Compare depth of neighbouring patches

- ###### Gaussian
    - Depths at higher scales are constrainted to be the average of lower scale
    - Different parameters for different rows 

- ###### Laplacian

# Relative Depth

1. ### Histogram 

    - 10 bin

#[Types of Monocular clues](http://web.mit.edu/sinhalab/Papers/sinha_top_down_NN.pdf)
    
1. texture variation
1. texture gradient
1. interposition
1. known object sizes
1. light and shading
1. haze
1. defocus 
