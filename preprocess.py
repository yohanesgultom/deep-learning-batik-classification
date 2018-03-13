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
    # get absolute path from arg
    mypath = sys.argv[1]
    dataset_file = sys.argv[2] if len(sys.argv) > 2 else DATASET_PATH
    index_file = sys.argv[3] if len(sys.argv) > 3 else DATASET_INDEX_PATH

    # iterate dir content
    stat = {}
    label_indexes = {}
    bar = progressbar.ProgressBar(maxval=700).start()
    count = 1
    i = 0

    # pytables file
    datafile = tables.open_file(dataset_file, mode='w')
    data = datafile.create_earray(datafile.root, 'data', tables.Float32Atom(shape=EXPECTED_DIM), (0,), 'batik')
    labels = datafile.create_earray(datafile.root, 'labels', tables.UInt8Atom(shape=(EXPECTED_CLASS)), (0,), 'batik')

    # iterate subfolders
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
                        img = normalize_and_filter(img)
                        # gather stat
                        stat[img.shape] = stat[img.shape] + 1 if img.shape in stat else 1
                        # for square in square_slice_generator(img, EXPECTED_SIZE):
                        #     # save train data
                        #     append_data_and_label(square, i, data, labels)
                        r = resize(img, EXPECTED_SIZE)
                        append_data_and_label(r, i, data, labels)
                        # update progress bar
                        bar.update(count)                        
                        count += 1
                    except Exception as err:
                        print(err)
                        print(path_sub)
                        sys.exit(0)

            i += 1
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
