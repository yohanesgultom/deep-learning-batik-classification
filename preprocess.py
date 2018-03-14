'''
Vectorize batik dataset

Author: yohanes.gultom@gmail.com

'''
import os
import sys
import cv2
import numpy as np
import json
import progressbar
import tables
import argparse
import imutils
from random import sample

# config
EXPECTED_MAX = 100.0
EXPECTED_MIN = -1 * EXPECTED_MAX
FILTER_THRESHOLD = -90.0
DATASET_PATH = 'dataset.h5'
DATASET_INDEX_PATH = 'dataset.index.json'

# global vars
EXPECTED_SIZE = 224
EXPECTED_CHANNELS = 3
# EXPECTED_DIM = (EXPECTED_CHANNELS, EXPECTED_SIZE, EXPECTED_SIZE)  # Theano
EXPECTED_DIM = (EXPECTED_SIZE, EXPECTED_SIZE, EXPECTED_CHANNELS)  # TensorFlow
EXPECTED_CLASS = 5
MAX_VALUE = 255
MEDIAN_VALUE = MAX_VALUE / 2.0


def square_slice_generator(data, size, slices_per_axis=5):
    if data.shape[0] <= size or data.shape[1] <= size:
        yield(resize(data, size))
    else:
        remaining_rows = data.shape[0] - size
        remaining_cols = data.shape[1] - size
        slide_delta_rows = remaining_rows / slices_per_axis
        slide_delta_cols = remaining_cols / slices_per_axis
        for i in range(slices_per_axis):
            row_start = i + i * slide_delta_rows
            row_end = row_start + size
            for j in range(slices_per_axis):
                col_start = j + j * slide_delta_cols
                col_end = col_start + size
                tmp = data[row_start:row_end, col_start:col_end]
                yield(tmp)


def resize(data, size):
    return cv2.resize(data, (size, size))

def zoomin(source, z):
	if z < 1:
		raise ValueError('z must be bigger than 1')
	if z == 1:
		return source

	resized = imutils.resize(source, width=int(round(z * source.shape[1])))	
	top_left = ((resized.shape[0] - source.shape[0]) / 2, (resized.shape[1] - source.shape[1]) / 2)
	cropped = resized[top_left[0]:(top_left[0] + source.shape[0]), top_left[1]:(top_left[1] + source.shape[1])]
	return cropped

def normalize_and_filter(data, expected_max=EXPECTED_MAX, median=MEDIAN_VALUE, threshold=FILTER_THRESHOLD):
    data = (data - median) / median * expected_max
    # data[data < threshold] = EXPECTED_MIN
    return data


def append_data_and_label(m, c, dataset, labels):
    # m = np.transpose(m, (2, 0, 1)) # Theano
    assert m.shape == EXPECTED_DIM
    dataset.append(np.array([m]))
    # one-hot encoding
    label = np.zeros(EXPECTED_CLASS)
    label[c] = 1.0
    assert label.shape == (EXPECTED_CLASS,)
    labels.append(np.array([label]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess and vectorize images dataset', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_path', help="Path to raw dataset directory")
    parser.add_argument('--classes_file', default=DATASET_INDEX_PATH, help="Output path for dataset classes index")
    parser.add_argument('--vector_file', default=DATASET_PATH, help="Output path for preprocessed and vectorized dataset")
    parser.add_argument('--rotate', '-r', metavar='ANGLE', type=int, default=0, help="Rotate images to certain angle (integer)")
    parser.add_argument('--zoomin', '-z', metavar='SCALE', type=float, default=1.0, help="Zoom in images to certain Scale (float)")

    args = parser.parse_args()
    mypath = args.dataset_path
    index_file = args.classes_file
    dataset_file = args.vector_file
    rotate = args.rotate
    scale = args.zoomin

    print(args)

    # iterate dir content
    stat = {}
    label_indexes = {}    
    count = 0
    i = 0

    # pytables file
    datafile = tables.open_file(dataset_file, mode='w')
    data = datafile.create_earray(datafile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'batik')
    labels = datafile.create_earray(datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'batik')

    # iterate subfolders
    num_dir = len([name for name in os.listdir(mypath)])
    bar = progressbar.ProgressBar(maxval=num_dir).start()
    for f in os.listdir(mypath):
        path = os.path.join(mypath, f)
        # exclude Mix motif
        if os.path.isdir(path) and f != 'Mix motif':
            label_indexes[i] = f
            for f_sub in os.listdir(path):
                path_sub = os.path.join(path, f_sub)                
                if os.path.isfile(path_sub):
                    try:
                        img = cv2.imread(path_sub)
                        # rotate
                        img = imutils.rotate(img, rotate) if rotate is not 0 else img
                        # scale
                        img = zoomin(img, scale) if scale > 1 else img
                        # normalize and filter
                        img = normalize_and_filter(img)
                        # gather stat
                        stat[img.shape] = stat[img.shape] + 1 if img.shape in stat else 1
                        r = resize(img, EXPECTED_SIZE)
                        append_data_and_label(r, i, data, labels)                        
                    except Exception as err:
                        print(err)
                        print(path_sub)
                        sys.exit(0)
            i += 1
        count += 1
        bar.update(count)
    bar.finish()

    print('{} records saved'.format(data.nrows))

    # write label index as json file
    with open(index_file, 'w') as f:
        json.dump(label_indexes, f)

    print((data.nrows,) + data[0].shape)
    print((labels.nrows,) + labels[0].shape)
    print(label_indexes)
    # print(stat)
    assert data[0].shape == EXPECTED_DIM
    assert labels[0].shape == (EXPECTED_CLASS,)

    # close file
    datafile.close()
