# Linear_Autoencoder_and_FFT_for_MNIST
Code and data from runs to find important features of MNIST data, both as is, 
and after FFT

## Quick Guide to the Repo

This repo contains tools for obtaining lower dimensional feature spaces
describing MNIST image data using autoencoders and Fourier transforms.

For a detailed description of how these results were obtained see X

### Directory structure

analysis:  
data:  
raw_data:  
src:  


## Background Information

### MNIST Image data

### DNN and CNN

Densely connected deep neural networks (DNN) have a large number of weights, 
making training difficult.  
A popular approach, therefore, to image recognition problems is to utilize 
convolutional neural networks (CNN).  CNN reduce the number of inputs

Therefore we hope that using a simple linear autoencoder, will allow us
to use DNN with a reasonable training burden.

### Autoencoders

An autoencoder is a neural network in which the number of neurons in each layer
decreases in the first half of the network.  The second half of the NN is a 
mirror image of the first half.  In particular the output layer has the same
number of neurons as the input layer.  (This number is also, generally, the
dimensionality of the feature space.) 

The NN is then trained so that the outputs of the network are as similar to 
the inputs as possible.  The reason for this is that then the first half of the
network is projecting the outputs into a lower dimensional parameter space, and
the second half of the network provides the mapping from this lower dimensional 
parameter space to the original feature space.

Thus autoencoders are used to find the lower dimensional feature space and
constituent features that best describe the original data.

An autoencoder is **linear** if the output from each neuron (after the input layer)
is the same as the input, i.e., if the activation function is the identity transformation.
Without loss of generality, linear autoencoders can be taken to have  
  
number_of_neurons in layer = [number_of_output_features, number_of_compressed_features, number_of_output_features]
  
as if the neurons between two sets of weights are linear, the result is simply multiplication
of the weight matrices before and after the given layer.  So a linear autoencoder with
multiple hidden layers would have a lot of completely unnecessary redundancy.

We first see how well we can map the MNIST image data using three different linear autoencoder architectures.
1.  Tied weights, no bias vectors
2.  Tied weights, bias vectors
3.  Untied weights.

An autoencoder has tied weights when weight matrix $L - i$ is the transopose of weight matrix $i$, 
assuming ... (need to be careful with labelling of layers, definition of $L$).

### Fourier Transformation

Fourier transformation maps between, e.g., spatial position data 

### Future Directions

As should be clear from the above my main goal in creating this repository is
to provide linear autoencoders for MNIST data to be used as inputs to DNN.
I have decided to make this a separate repository, as one can go in other
directions as well using the autoencoder data.

A particularly interesting potential use of the reduced dimensional feature space
from the autoencoders is to cluster in a higher (but not 784) dimensional 
parameter space.  This should provide an interesting geometric perspective on
which samples are misclassified.

Future work may consider more powerful, non-linear, and possibly deep autoencoders.
I have started with the linear case both because it is simpler and because
I wanted to clearly separate the deep learning part of the project from the
autoencoder part.


