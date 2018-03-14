'''
Evaluate batik MLP softmax classifier

Author: yohanes.gultom@gmail.com
'''
from __future__ import print_function
import numpy as np
import sys
import tables
import argparse
import json
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.models import model_from_json

# default training parameters
CV = 7  
BATCH_SIZE = 700 # fine on GTX 980 4 GB
NB_EPOCH = 100
MODEL_JSON = 'model.json'
MODEL_WEIGHTS = 'model.weights.h5'

# const
# FEATURES_DIM = (512, 7, 7) # Theano
FEATURES_DIM = (7, 7, 512) # TensorFlow
EXPECTED_CLASS = 5


def create_model_tanh_two_layers(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model


def build_model(input_shape, num_class, X_train, y_train, X_test, y_test):
    model = create_model_tanh_two_layers(input_shape, num_class)
    model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, y_test),
        shuffle=True)

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def save_model(model, model_json_file, weights_h5_file):
    json_string = model.to_json()
    with open(model_json_file, 'w') as f:
        json.dump(json_string, f)
    model.save_weights(weights_h5_file)


def load_model(model_json_file, weights_h5_file):
    model = None
    with open(model_json_file, 'r') as f:
        json_string = f.read()  
        model = model_from_json(json_string)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate neural network classifiers using extracted dataset features', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('test_file', help="Path to test data (features) input file")
    parser.add_argument('--train_file', help="Path to train data (features) input file")
    parser.add_argument('--val_file', help="Path to validation data (features) input file")
    parser.add_argument('--model_json', help="Path to JSON model structure input file")
    parser.add_argument('--model_weights', help="Path to model weights input file")
    parser.add_argument('--n_folds', type=int, default=CV, help="Number of folds (K) for K-fold cross validation")
    parser.add_argument('--nb_epoch', type=int, default=NB_EPOCH, help="Training epoch")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Training batch size")

    args = parser.parse_args()
    test_file = args.test_file
    train_file = args.train_file
    val_file = args.train_file
    model_json = args.model_json
    model_weights = args.model_weights
    n_folds = args.n_folds
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size

    print('batch_size: {}'.format(batch_size))
    print('nb_epoch: {}'.format(nb_epoch))

    input_dim = (np.prod(FEATURES_DIM),)

    print('Loading validation dataset: {}'.format(val_file))
    val_datafile = tables.open_file(val_file, mode='r')
    val_dataset = val_datafile.root
    print('Validation data: {}'.format((val_dataset.data.nrows,) + val_dataset.data[0].shape))

    model = None
    if model_json is not None and model_weights is not None:
        print('Loading model structure: {}'.format(model_json))
        print('Loading model weights: {}'.format(model_weights))
        model = load_model(model_json, model_weights)

    elif train_file is not None:
        print('Loading train dataset: {}'.format(train_file))
        train_datafile = tables.open_file(train_file, mode='r')
        train_dataset = train_datafile.root
        print('Train data: {}'.format((train_dataset.data.nrows,) + train_dataset.data[0].shape))

        model = build_model(
            input_dim, 
            EXPECTED_CLASS, 
            train_dataset.data[:], 
            train_dataset.labels[:], 
            val_dataset.data[:], 
            val_dataset.labels[:], 
        )

        print('Saving model structure: {}'.format(MODEL_JSON))
        print('Saving model weights: {}'.format(MODEL_WEIGHTS))
        save_model(model, MODEL_JSON, MODEL_WEIGHTS)

    else:
        raise ValueError('No saved model or training data provided')
      
    print('Loading test dataset: {}'.format(test_file))
    test_datafile = tables.open_file(test_file, mode='r')
    test_dataset = test_datafile.root
    print('Test data: {}'.format((test_dataset.data.nrows,) + test_dataset.data[0].shape))

    X_test = test_dataset.data[:]
    y_test = test_dataset.labels[:]
    y_pred = model.predict(X_test)
    predictions = y_pred.argmax(1)
    truths = y_test.argmax(1)
    score = accuracy_score(truths, predictions)

    print("Accuracy: {}".format(score))