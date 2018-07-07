""" 
Module with methods for training autoencoders to obtain good autoencodings
to compress features.
"""

import numpy as np
import os
import sys
import tensorflow as tf

# Make sure that we are running Python 3
assert sys.version_info[0] == 3 # Need to produce Python 3 pickles.

repo_head_dir = os.path.dirname(os.path.dirname(__file__))
python_dir = os.path.join(repo_head_dir, 'autoencoder_fft_mnist')

def init_weights(shape, stddev = 0.1):
    init_random_dist = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init_random_dist)

def basic_linear_auto_encoder_arch(num_initial_features, 
	num_compressed_features, input_placeholder):
    """ Architecture for autoencoder with tied weights and no bias vectors. """
    X = input_placeholder
    W = init_weights([num_initial_features, num_compressed_features]) 

    return {'autoencoder': tf.matmul((tf.matmul(X,W)), tf.transpose(W)),
            'W': W}

def linear_auto_encoder_arch(num_initial_features, num_compressed_features, input_placeholder):
    """ Architecture for autoencoder with tied weights and bias vectors. """
    X = input_placeholder
    W = init_weights([num_initial_features, num_compressed_features]) 
    b1 = init_weights([num_compressed_features])
    b2 = init_weights([num_initial_features])

    return {'autoencoder': tf.matmul((tf.matmul(X,W) + b1), tf.transpose(W)) + b2,
            'W': W,
            'b1': b1,
            'b2': b2}

def untied_linear_auto_encoder_arch(num_initial_features,
    num_compressed_features, input_placeholder):
    """ Architecture for autoencoder with untied weights and bias vectors. """
    X = input_placeholder
    W1 = init_weights([num_initial_features, num_compressed_features])
    W2 = init_weights([num_compressed_features, num_initial_features])
    b1 = init_weights([num_compressed_features])
    b2 = init_weights([num_initial_features])

    return {'autoencoder': tf.matmul((tf.matmul(X,W1) + b1), W2) + b2,
            'W1': W1,
            'W2': W2,
            'b1': b1,
            'b2': b2}

def next_batch(X_data, batch_size):
    indices = np.random.randint(0, len(X_data), size = batch_size)
    return X_data[indices]

