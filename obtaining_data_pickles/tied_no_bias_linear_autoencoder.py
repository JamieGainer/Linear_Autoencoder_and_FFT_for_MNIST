""" This code is for tied linear autoencoders without bias terms.
	Call with

	python tied_no_bias_linear_autoencoder [num_compressed_features num_steps \
	batch_size tf_seed np_seed learning_algorithm learning_rate]

"""

import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time

import autoencoder_obtainer as auto

# Start (wall clock time) timer
start = time.time()

# Obtain parameters for run

parameter = {             # default parameters
	'num_compressed_features': 30,
	'num_steps': 100000,
	'batch_size': 16,
	'tf_seed': 11235,
	'np_seed': 842,
	'learning_algorithm': 'sgd',
	'learning_rate': 0.5
}

for index, name, description, type_name in zip([1, 2, 3, 4, 5, 7],
	['num_compressed_features', 'num_steps', 'batch_size', 'tf_seed', 
	 'np_seed', 'learning_rate'],
	['number of compressed features', 'number of steps', 'batch size', 
	'tensorflow random number seed', 'numpy random number seed', 
	'learning_rate'],
	[int, int, int, int, int, float]):
	
	if len(sys.argv) > index:
		try:
			parameter[name] = type_name(sys.argv[index])
		except ValueError:
			error_string = 'Argument ' + str(index) + ', ' +  description
			error_string += ', must be of ' + str(type_name) + '.'
			raise ValueError(error_string)


if len(sys.argv) > 6:
	parameter['learning_algorithm'] = sys.argv[6]

if parameter['learning_algorithm'] not in ['sgd', 'adam']:
	raise ValueError('Argument 5, Learning Algorithm must be "sgd" or "adam".')

# set seeds

tf.set_random_seed(parameter['tf_seed'])
np.random.seed(seed=parameter['np_seed'])


# obtain MNIST data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
parameter['num_initial_features'] = 784
assert mnist.train.images.shape[1] == parameter['num_initial_features']
assert len(mnist.train.images.shape) == 2


# Prepare for tensor flow session

X = tf.placeholder(tf.float32,shape=[None, parameter['num_initial_features']])
y_dict = auto.basic_linear_auto_encoder_arch(parameter['num_initial_features'], 
	parameter['num_compressed_features'], X)
y = y_dict['autoencoder']

loss_function = tf.reduce_mean((y - X)**2)
if parameter['learning_algorithm'] == 'sgd':
	optimizer = tf.train.GradientDescentOptimizer(
		learning_rate = parameter['learning_rate'])
else:
	optimizer = tf.train.AdamOptimizer()

train = optimizer.minimize(loss_function)

init = tf.global_variables_initializer()


# Tensor flow session.  Keep track of loss parameter for learning curve data

parameter['step_numbers'] = []
parameter['loss_function_values'] = []

with tf.Session() as sess:
    
    sess.run(init)
    
    for step in range(parameter['num_steps']):
        
        if (((step < 10) or (step < 100 and step % 10 == 0) or 
        	(step < 1000 and step % 100 == 0) or
        	(step < 10000 and step % 1000 == 0) or
        	(step % 10000 == 0))  or
            step == parameter['num_steps'] - 1):
        	parameter['step_numbers'].append(step)
        	parameter['loss_function_values'].append(sess.run(loss_function, 
        		feed_dict = {X: mnist.train.images}))
        	print('On step', step)
        
        batch_x = auto.next_batch(mnist.train.images, parameter['batch_size'])
        
        sess.run(train, feed_dict={X: batch_x})
    
    parameter['W'] = sess.run(y_dict['W'])

parameter['time'] = time.time() - start

# Write data to pickle

def pickle_file_name(num):
	return 'run-' + str(pickle_file_label) + '.pickle'

pickle_file_label = 1
while os.path.exists(pickle_file_name(pickle_file_label)):
	pickle_file_label += 1
file_name = pickle_file_name(pickle_file_label)

with open(file_name, 'wb') as pickle_file:
	pickle.dump(parameter, pickle_file)

