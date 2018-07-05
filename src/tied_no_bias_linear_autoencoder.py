""" Google compliant header """

import numpy as np
import tensorflow as tf

def init_weights(shape, stddev = 0.1):
    init_random_dist = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(init_random_dist)

def basic_linear_auto_encoder_arch(num_initial_features, 
	num_compressed_features, input_placeholder):

    X = input_placeholder
    W = init_weights([num_initial_features, num_compressed_features]) 

    return {'autoencoder': tf.matmul((tf.matmul(X,W)), tf.transpose(W)),
            'W': W}

def linear_auto_encoder_arch(num_initial_features, num_compressed_features, input_placeholder):

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
