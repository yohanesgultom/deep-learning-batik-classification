'''
Evaluate batik MLP softmax classifier

Author: yohanes.gultom@gmail.com
'''
from __future__ import print_function
import numpy as np
import sys
import tables
import argparse
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.models import load_model
from sklearn.metrics import accuracy_score

# default training parameters
CV = 7  
BATCH_SIZE = 700 # fine on GTX 980 4 GB
NB_EPOCH = 100
MODEL_FILE = 'model.h5'

# const
# FEATURES_DIM = (512, 7, 7) # Theano
FEATURES_DIM = (7, 7, 512) # TensorFlow
EXPECTED_CLASS = 5

def create_model_relu_tanh_two_layers(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model

def create_model_tanh_two_layers(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model

def create_model_sklearn(input_shape, num_class):
    model = Sequential()
    model.add(Dense(FEATURES_DIM[2], activation='relu', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model

def build_model(input_shape, num_class, X_train, y_train, X_test, y_test, nb_epoch):
    model = create_model_sklearn(input_shape, num_class)  

    # set optimization function
    # optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  
    model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_data=(X_test, y_test),
        shuffle=True)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate neural network classifiers using extracted dataset features', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('test_file', help="Path to test data (features) input file")
    parser.add_argument('--train_file', help="Path to train data (features) input file")
    parser.add_argument('--val_file', help="Path to validation data (features) input file")
    parser.add_argument('--model_file', help="Path to complete model input file")
    parser.add_argument('--n_folds', type=int, default=CV, help="Number of folds (K) for K-fold cross validation")
    parser.add_argument('--nb_epoch', type=int, default=NB_EPOCH, help="Training epoch")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help="Training batch size")

    args = parser.parse_args()
    test_file = args.test_file
    train_file = args.train_file
    val_file = args.val_file
    model_file = args.model_file
    n_folds = args.n_folds
    nb_epoch = args.nb_epoch
    batch_size = args.batch_size

    print('batch_size: {}'.format(batch_size))
    print('nb_epoch: {}'.format(nb_epoch))

    input_dim = (FEATURES_DIM[2],)

    model = None
    if model_file is not None:
        print('Loading model: {}'.format(model_file))
        model = load_model(model_file)

    elif train_file is not None:
        print('Loading train dataset: {}'.format(train_file))
        train_datafile = tables.open_file(train_file, mode='r')
        train_dataset = train_datafile.root
        print('Train data: {}'.format((train_dataset.data.nrows,) + train_dataset.data[0].shape))

        print('Loading validation dataset: {}'.format(val_file))
        val_datafile = tables.open_file(val_file, mode='r')
        val_dataset = val_datafile.root
        print('Validation data: {}'.format((val_dataset.data.nrows,) + val_dataset.data[0].shape))

        model = build_model(
            input_dim, 
            EXPECTED_CLASS, 
            train_dataset.data[:], 
            train_dataset.labels[:], 
            val_dataset.data[:], 
            val_dataset.labels[:], 
            nb_epoch,
        )

        print('Saving model: {}'.format(MODEL_FILE))
        model.save(MODEL_FILE)

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