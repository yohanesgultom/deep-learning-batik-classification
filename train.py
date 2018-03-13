'''
Train batik MLP softmax classifier

Author: yohanes.gultom@gmail.com
'''
from __future__ import print_function
import numpy as np
import sys
import tables
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from keras import backend as K

# training parameters
CV = 7  
BATCH_SIZE = 200
NB_EPOCH = 300

# const
# FEATURES_DIM = (512, 7, 7) # Theano
FEATURES_DIM = (7, 7, 512) # TensorFlow
EXPECTED_CLASS = 5

def create_model_relu_two_layers(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='relu'))
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

def create_model_sigmoid_two_layers(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='sigmoid', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model


def create_model_relu_tanh_two_layers(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model

def create_model_relu_one_layer(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model


def create_model_tanh_one_layer(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model


def create_model_sigmoid_one_layer(input_shape, num_class):
    model = Sequential()
    model.add(Dense(4096, activation='sigmoid', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(num_class, activation='softmax', init='uniform'))
    return model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train,
        batch_size=BATCH_SIZE,
        nb_epoch=NB_EPOCH,
        validation_data=(X_test, y_test),
        shuffle=True)

    y_pred = model.predict_on_batch(X_test)
    predictions = y_pred.argmax(1)
    truths = y_test.argmax(1)
    return accuracy_score(truths, predictions)


if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    n_folds = int(sys.argv[3]) if len(sys.argv) > 3 else CV

    print('BATCH_SIZE: {}'.format(BATCH_SIZE))
    print('NB_EPOCH: {}'.format(NB_EPOCH))

    # loading dataset
    print('Loading train dataset: {}'.format(train_file))
    train_datafile = tables.open_file(train_file, mode='r')
    train_dataset = train_datafile.root
    print('Train data: {}'.format((train_dataset.data.nrows,) + train_dataset.data[0].shape))

    print('Loading test dataset: {}'.format(test_file))
    test_datafile = tables.open_file(test_file, mode='r')
    test_dataset = test_datafile.root
    print('Test data: {}'.format((test_dataset.data.nrows,) + test_dataset.data[0].shape))
      
    X = np.concatenate((train_dataset.data[:], test_dataset.data[:]), axis=0)
    y = np.concatenate((train_dataset.labels[:], test_dataset.labels[:]), axis=0)

    # close dataset
    train_datafile.close()
    test_datafile.close()

    model_creators = [
        create_model_sigmoid_two_layers,
        create_model_relu_two_layers,
        create_model_tanh_two_layers,
        create_model_relu_tanh_two_layers,
        create_model_sigmoid_one_layer,
        create_model_relu_one_layer,
        create_model_tanh_one_layer,
    ]

    results = []
    input_dim = (np.prod(FEATURES_DIM),)
    for j, create_model in enumerate(model_creators):
        skf = StratifiedKFold(y.argmax(1), n_folds=n_folds, shuffle=True)
        scores = []
        for i, (train, test) in enumerate(skf):
            print("Running fold {}/{}".format(i+1, n_folds))
            model = None # Clearing the NN.
            sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
            model = create_model(input_dim, EXPECTED_CLASS)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
            score = train_and_evaluate_model(model, X[train], y[train], X[test], y[test])
            scores.append(score)
            K.clear_session()

        results.append({
            'name': create_model.func_name,
            'scores': np.array(scores)
        })

    # print results
    print('\n')
    for result in results:
        name = result['name']
        mean = result['scores'].mean()
        stdev = result['scores'].std() * 2
        print("{} CV accuracy: {:0.2f} (+/- {:0.2f})".format(name, mean, stdev))

