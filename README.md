# Linear_Autoencoder_and_FFT_for_MNIST

This repo contains tools for obtaining lower dimensional feature spaces
describing MNIST image data using autoencoders.  I am interested
in uses for fourier transformed data, therefore this repo includes
tools for generating

1. autoencoding of 784 (= 28 x 28 black and white pixel) MNIST data
2. autoencoding of 784 (= 28 x 28 black and white pixel) Fast Fourier Transformed (FFT) MNIST data
3. feature spaces consisting of $m$ features from autoencoding MNIST data and $n$
features from autoencoding FFT MNIST data.

The autoencoders here are all linear autoencoders with no bias vectors.  
Future development may include non-linear autoencoders, though
for many of my anticipated applications, it makes more sense to do the
hard work in the latter stages and keep obtaining the lower dimensional
parameter space simple.

All "user facing" code is available in autoencoder_fft_mnist/fft_autoencoder.py.
It requires Python 3.  This code allows one to obtain the autoencoded MNIST data described above.
It also contains methods for mapping back to the original MNIST feature space
for visualization, etc.

The autoencoders (both of MNIST image data and FFT-ed 
MNIST image data), as well as data about how they were obtained, are stored
in the data_pickles directory.  

Code used to obtain these pickles and a
notebook exploring them is found in the obtaining_data_pickles directory.

Finally a notebook in the images directory can be used to visualize the
effects of the various autoencodings.

Anticpated uses include having lower dimensional spaces for deep densely connected 
neural networks, as well as exploring the geometry of the data using $k$-means clustering,
etc.  I will list my repos that explore these areas as I create them.

Ultimately, of course, the goal is to learn about and to develop new approaches
to image classification beyond the MNIST data set.