"""
Header
"""

import numpy as np
import os
import pickle
import sys
import tensorflow as tf

assert sys.version_info[0] == 3 # We need to read python3-style pickles for most module functionality

path = os.path.abspath(__file__)
dir_path = os.path.dirname(os.path.dirname(path))
pickle_dir = os.path.join(dir_path, 'data_pickles')
autoencoder_dir = os.path.join(pickle_dir, 'tied_no_bias')
fft_autoencoder_dir = os.path.join(pickle_dir, 'fft_tied_no_bias')
plots_dir = os.path.join(dir_path, 'plots')


def compress_fft(fft_data, data_set = 'mnist'):
	""" For now just works with MNIST data set """
	assert data_set == 'mnist'
    
	temp_array = fft_data[:,:,:15].reshape(-1,28 * 15)
	return_array = np.zeros((len(temp_array), 2 * temp_array.shape[1]))
	return_array[:,::2] = np.real(temp_array)
	return_array[:,1::2] = np.imag(temp_array)
	mask = (np.ones_like(return_array[0]) == np.ones_like(return_array[0]))
	mask[30 * np.arange(15,28)] = False
	mask[30 * np.arange(15,28) + 1] = False
	mask[28 + 30 * np.arange(15,28)] = False
	mask[28 + 30 * np.arange(15,28) + 1] = False
	#
	mask[1] = False
	mask[29] = False
	mask[14 * 30 + 1] = False
	mask[14 * 30 + 29] = False
	return return_array[:,mask]


def decompress_fft(compressed_fft_data, data_set = 'mnist'):
	""" For now just works with MNIST data set """
	assert data_set == 'mnist'
    
	step_one = np.zeros((len(compressed_fft_data), 840))
	step_one[:,0] = compressed_fft_data[:,0]
	step_one[:,2:29] = compressed_fft_data[:,1:28]
	step_one[:,30:421] = compressed_fft_data[:,28:419]
	step_one[:,422:449] = compressed_fft_data[:,419:446]
	#
	step_one[:,[30 * np.arange(15,28)]] = step_one[:,[30 * np.arange(13,0,-1)]]
	step_one[:,[30 * np.arange(15,28) + 1]] = -step_one[:,[30 * np.arange(13,0,-1) + 1]]
	step_one[:,[28 + 30 * np.arange(15,28)]] = step_one[:,[28 + 30 * np.arange(13,0,-1)]]
	step_one[:,[28 + 30 * np.arange(15,28) + 1]] = -step_one[:,[28 + 30 * np.arange(13,0,-1) + 1]]
	#
	step_one_index, compressed_index = 452, 446
	for i in range(13):
		a, b = step_one_index, step_one_index + 26
		c, d = compressed_index, compressed_index + 26
		step_one[:, a:b] = compressed_fft_data[:, c:d]
		step_one_index = b + 4
		compressed_index = d

	step_two = np.zeros((len(compressed_fft_data), 28, 28), dtype=np.complex128)
	real_ar, im_ar = (step_one.reshape(-1,28,30)[:,:,::2],
		step_one.reshape(-1,28,30)[:,:,1::2])
	step_two[:, :, :15] += real_ar + 1j * im_ar  
	step_two[:, 1:, 15:] = np.conjugate(step_two[:,:0:-1,13:0:-1])
	step_two[:,0,15:] = np.conjugate(step_two[:,0,13:0:-1])

	return step_two
	
# Obtained "compressed" Fourier transformed versions of the MNIST image sets

def compressed_fft_data(mnist_images):
	return compress_fft(np.fft.fft2(mnist_images.reshape(-1,28,28), axes = [1,2]))


def add_compressed_fft_data(mnist_data_set):
	mnist_data_set.fft_images = compressed_fft_data(mnist_data_set.images)


def add_autoencoded_mnist_data_set(mnist_data_set, dimension):
	""" Add data to mnist data set (train, validation, or test)
		that has been Fast Fourier Transformed and then passed
		to an autoencoder of the specified dimension """

	pickle_name = 'nc-' + str(dimension) + '.pickle'
	
	if pickle_name not in os.listdir(autoencoder_dir):
		raise ValueError('No autoencoder for image data found for ' 
			+ str(dimension) + ' dimensions.')

	pickle_pathname = os.path.join(autoencoder_dir, pickle_name)
	
	with open(pickle_pathname, 'rb') as pickle_file:
		d = pickle.load(pickle_file)
		W = d['W']

	encoded_data = mnist_data_set.images.dot(W)

	if 'autoencoder' in dir(mnist_data_set):
		mnist_data_set.autoencoder[dimension] = encoded_data
	else:
		mnist_data_set.autoencoder = {dimension: encoded_data}


def add_fft_autoencoded_mnist_data_set(mnist_data_set, dimension):
	""" Add data to mnist data set (train, validation, or test)
		that has been Fast Fourier Transformed and then passed
		to an autoencoder of the specified dimension """

	pickle_name = 'nc-' + str(dimension) + '.pickle'
	
	if pickle_name not in os.listdir(fft_autoencoder_dir):
		raise ValueError('No autoencoder for FFT data found for ' 
			+ str(dimension) + ' dimensions.')

	pickle_pathname = os.path.join(fft_autoencoder_dir, pickle_name)
	
	with open(pickle_pathname, 'rb') as pickle_file:
		d = pickle.load(pickle_file)
		W = d['W']

	if 'fft_images' in dir(mnist_data_set):
		fft_data = mnist_data_set.fft_images
	else:
		fft_data = compressed_fft_data(mnist_data_set.images)

	encoded_data = fft_data.dot(W)

	if 'fft_autoencoder' in dir(mnist_data_set):
		mnist_data_set.fft_autoencoder[dimension] = encoded_data
	else:
		mnist_data_set.fft_autoencoder = {dimension: encoded_data}


def add_hybrid_autoencoded_mnist_data_set(mnist_data_set, dim_tuple):
	""" Add data to a mnist data set (train, validation, or test)
	consisting of the original data autoencoded to dim_tuple[0] dimensions
	and the FFT data autoencoded to dim_tuple[1] dimensions.  The result
	is a numpy array with dim_tuple[0] + dim_tuple[1] features. """
		
	if dim_tuple == (0,0):
		return

	if dim_tuple[0] == 0:
		if ('fft_autoencoder' not in dir(mnist_data_set) or 
			dim_tuple[1] not in mnist_data_set.fft_autoencoder):
			add_fft_autoencoded_mnist_data_set(mnist_data_set, dim_tuple[1])
		hybrid_data = mnist_data_set.fft_autoencoder[dim_tuple[1]]
	elif dim_tuple[1] == 0:
		if ('autoencoder' not in dir(mnist_data_set) or 
			dim_tuple[0] not in mnist_data_set.autoencoder):
			add_autoencoded_mnist_data_set(mnist_data_set, dim_tuple[0])
		hybrid_data = mnist_data_set.autoencoder[dim_tuple[0]]
	else:
		if not ('autoencoder' in dir(mnist_data_set) and 
			dim_tuple[0] in mnist_data_set.autoencoder):
			add_autoencoded_mnist_data_set(mnist_data_set, dim_tuple[0])
		if not ('fft_autoencoder' in dir(mnist_data_set) and 
			dim_tuple[1] in mnist_data_set.fft_autoencoder):
			add_fft_autoencoded_mnist_data_set(mnist_data_set, dim_tuple[1])
		hybrid_data = np.concatenate((mnist_data_set.autoencoder[dim_tuple[0]],
								  mnist_data_set.fft_autoencoder[dim_tuple[1]]),
								  axis = 1)

	if 'hybrid_autoencoder' in dir(mnist_data_set):
		mnist_data_set.hybrid_autoencoder[dim_tuple] = hybrid_data
	else:
		mnist_data_set.hybrid_autoencoder = {dim_tuple: hybrid_data}


def decode(encoded_data, dimension):
	""" Map from the autoencoded image space (with specified dimension) to the 
	original image feature space """
	pickle_name = 'nc-' + str(dimension) + '.pickle'
	
	if pickle_name not in os.listdir(autoencoder_dir):
		raise ValueError('No autoencoder for image data found for ' 
			+ str(dimension) + ' dimensions.')

	pickle_pathname = os.path.join(autoencoder_dir, pickle_name)
	
	with open(pickle_pathname, 'rb') as pickle_file:
		d = pickle.load(pickle_file)
		W = d['W']

	return encoded_data.dot(W.transpose())


def fft_decode(encoded_data, dimension):
	""" Map from the autoencoded FFT space (with specified dimension) to the 
	"compressed" FFT feature space """
	pickle_name = 'nc-' + str(dimension) + '.pickle'
	
	if pickle_name not in os.listdir(fft_autoencoder_dir):
		raise ValueError('No autoencoder for image data found for ' 
			+ str(dimension) + ' dimensions.')

	pickle_pathname = os.path.join(fft_autoencoder_dir, pickle_name)
	
	with open(pickle_pathname, 'rb') as pickle_file:
		d = pickle.load(pickle_file)
		W = d['W']

	return encoded_data.dot(W.transpose())


def image_from_compressed_fft(autoencoded_fft_data, data_set = 'mnist'):
	""" Obtain image pixels from "compressed" Fourier transformed MNIST data """
	dimension = autoencoded_fft_data.shape[1]
	compressed_fft_data = fft_decode(autoencoded_fft_data, dimension)
	return np.real(np.fft.ifft2(decompress_fft(compressed_fft_data))).reshape(-1, 784)


def hybrid_decode(encoded_data, dim_tuple):
	"""Obtain image pixels from a hybrid (autoencoded and FFT autoencoded)
	feature space"""
	if dim_tuple[0] == 0:
		return image_from_compressed_fft(encoded_data)
	if dim_tuple[1] == 0:
		return decode(encoded_data, dim_tuple[0])

	autoencoded_data = encoded_data[:,:dim_tuple[0]]
	fft_autoencoded_data = encoded_data[:, dim_tuple[0]:]

	autoencoded_image = decode(autoencoded_data, dim_tuple[0])
	fft_autoencoded_image = image_from_compressed_fft(
		fft_autoencoded_data, dim_tuple[1])

	return (dim_tuple[0] * autoencoded_image + 
		dim_tuple[1] * fft_autoencoded_image)/(
		dim_tuple[0] + dim_tuple[1])