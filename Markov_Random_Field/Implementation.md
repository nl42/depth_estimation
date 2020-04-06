# Notes

- ###### Depth represented on log scale
- ###### Laplasian outperforms Gaussian
- ###### Many cues were given large weights
    - Training requires sufficient cues

# Depth Features

1. ### texture variation
    - extracted from image intensity:
        - using Laws' masks      
1. ### texture gradient
    - Combine:
        - intensity channel 
        - six oriented edge filters
1. ### haze
    - local averaging filter:
        - colour channels
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
