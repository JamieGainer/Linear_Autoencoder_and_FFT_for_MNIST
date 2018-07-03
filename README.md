# Linear_Autoencoder_and_FFT_for_MNIST
Code and data from runs to find important features of MNIST data, both as is, 
and after FFT

## DNN and CNN

Densely connected deep neural networks (DNN) have a large number of weights, 
making training difficult.  
A popular approach, therefore, to image recognition problems is to utilize 
convolutional neural networks (CNN).  CNN reduce the number of inputs


Therefore we hope that using a simple linear autoencoder, will allow us
to use DNN with a reasonable training burden.

## Autoencoders

An autoencoder is a neural network in which the number of 

## Fourier Transformation

## Future Directions

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


