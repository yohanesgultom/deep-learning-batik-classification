'''
Batik VGG16 feature extractor

Author: yohanes.gultom@gmail.com
'''

import os
import tables
import sys
import argparse
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Flatten


# params
BATCH_SIZE = 700 # fine on GTX 980

# const
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
# EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)  # Theano
EXPECTED_DIM = (EXPECTED_SIZE, EXPECTED_SIZE, EXPECTED_CHANNELS)  # Tensorflow
FEATURES_FILE = 'features.h5'
# FEATURES_DIM = (512, 7, 7) # Theano
FEATURES_DIM = (7, 7, 512) # TensorFlow
EXPECTED_CLASS = 5


def build_extractor(input_shape):
	vgg16 = VGG16(weights='imagenet', include_top=False, pooling='avg')
	input = Input(shape=input_shape, name='input')
	output = vgg16(input)
	return Model(inputs=input, outputs=output)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extract features from vector dataset using VGG16', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('dataset_file', help="Path to vector dataset input file")
	parser.add_argument('--features_file', default=FEATURES_FILE, help="Output path for features file")
	parser.add_argument('--batch_size', default=BATCH_SIZE, help="Fit batch size")

	args = parser.parse_args()
	dataset_file = args.dataset_file
	features_file = args.features_file
	batch_size = args.batch_size

	# loading dataset
	print('Loading preprocessed dataset: {}'.format(dataset_file))
	datafile = tables.open_file(dataset_file, mode='r')
	dataset = datafile.root
	print((dataset.data.nrows,) + dataset.data[0].shape)

	# feature extractor    
	extractor = build_extractor(EXPECTED_DIM)

	print('Feature extraction')
	flatten_dim = (FEATURES_DIM[2],)
	features_datafile = tables.open_file(features_file, mode='w')
	features_data = features_datafile.create_earray(features_datafile.root, 'data', tables.Float32Atom(shape=flatten_dim), (0,), 'dream')
	features_labels = features_datafile.create_earray(features_datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'dream')

	i = 0
	while i < dataset.data.nrows:
		end = i + batch_size
		data_chunk = dataset.data[i:end]
		label_chunk = dataset.labels[i:end]
		i = end
		features = extractor.predict(data_chunk, verbose=1)
		features_data.append(features)
		features_labels.append(label_chunk)

	assert features_datafile.root.data.nrows == dataset.data.nrows
	assert features_datafile.root.labels.nrows == dataset.labels.nrows

	# close feature file
	features_datafile.close()
