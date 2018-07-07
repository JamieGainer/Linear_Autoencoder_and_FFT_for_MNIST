"""
Some tests of the installation, mostly to make sure that the
appropriate data pickles can be found and read.
"""

import os
import pickle
from fft_autoencoder import pickle_dir

def tests():
	expected_dirs = ['tied_no_bias', 'tied_bias', 'untied',
					 'learning_rates', 'fft_tied_no_bias']    

	for expected_dir in expected_dirs:
		assert expected_dir in os.listdir(pickle_dir)

	print('All expcted pickle directories found.')

	for dir_name, subdirs, file_names in os.walk(pickle_dir):
		if 'tied' not in dir_name:
			continue
		for file_name in file_names:
			if 'nc-' in file_name and '.pickle' in file_name:
				pathname = os.path.join(dir_name, file_name)
				nc = int(file_name.replace('nc-', '').replace('.pickle', ''))
				with open(pathname, 'rb') as pickle_file:
					d = pickle.load(pickle_file)
					assert d['num_compressed_features'] == nc

	print('Autoencoders found in correct pickles.')

tests()
