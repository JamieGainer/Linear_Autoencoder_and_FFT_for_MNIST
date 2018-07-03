# Linear_Autoencoder_and_FFT_for_MNIST
Code and data from runs to find important features of MNIST data, both as is, 
and after FFT

## Quick Guide to the Repo



## Background Information

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

In particular, we will be 

### Fourier Transformation

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


