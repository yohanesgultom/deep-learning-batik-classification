"""
Helper methods for SIFT and SURF Bag of Words feature extractor

Author: yohanes.gultom@gmail.com
"""

import cv2
import numpy as np
import os
import sys
import progressbar
from sklearn.cluster import MiniBatchKMeans

def get_dir_info(dir_path):
    dir_names = os.listdir(dir_path)
    file_paths = []
    file_dir_indexes = []

    for i in range(len(dir_names)):
        p = dir_names[i]
        subdir = os.path.join(dir_path, p)
        files = os.listdir(subdir)
        for f in files:
            file_paths.append(os.path.join(subdir, f))
            file_dir_indexes.append(i)
    
    return dir_names, file_paths, file_dir_indexes

def build_extractor(xfeatures2d, dir_names=None, file_paths=None, dictionary_size=None, dictionary=None):
    if dir_names is not None and file_paths is not None:
        print('Computing descriptors..')        
        BOW = cv2.BOWKMeansTrainer(dictionary_size)
        num_files = len(file_paths)
        bar = progressbar.ProgressBar(maxval=num_files).start()
        for i in range(num_files):
            p = file_paths[i]
            image = cv2.imread(p)
            gray = cv2.cvtColor(image, cv2.cv2.IMREAD_GRAYSCALE)
            kp, dsc = xfeatures2d.detectAndCompute(gray, None)
            BOW.add(dsc)
            bar.update(i)
        bar.finish()

        print('Creating BoW vocabs using K-Means clustering with k={}..'.format(dictionary_size))
        dictionary = BOW.cluster()

    if dictionary is not None:        
        print "bow dictionary", np.shape(dictionary)
        extractor = cv2.BOWImgDescriptorExtractor(xfeatures2d, cv2.BFMatcher(cv2.NORM_L2))
        extractor.setVocabulary(dictionary)
    return extractor, dictionary


def build_dataset(dir_names, file_paths, file_dir_indexes, extractor, xfeatures2d):
    assert len(file_paths) == len(file_dir_indexes)    
    X = [] # data
    y = file_dir_indexes # labels

    print('Computing features using Bag-of-Words dictionary..')
    num_files = len(file_paths)
    bar = progressbar.ProgressBar(maxval=num_files).start()
    for i in range(num_files):
        p = file_paths[i]
        im = cv2.imread(p, 1)
        gray = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)
        feature = extractor.compute(gray, xfeatures2d.detect(gray))
        X.extend(feature)
        bar.update(i)
    bar.finish()
    return X, y


def build_dictionary(xfeatures2d, dir_names, file_paths, dictionary_size):
    print('Computing descriptors..')        
    desc_list = []
    num_files = len(file_paths)
    bar = progressbar.ProgressBar(maxval=num_files).start()
    for i in range(num_files):
        p = file_paths[i]
        image = cv2.imread(p)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, dsc = xfeatures2d.detectAndCompute(gray, None)
        desc_list.extend(dsc)
        bar.update(i)
    bar.finish()

    print('Creating BoW dictionary using K-Means clustering with k={}..'.format(dictionary_size))
    dictionary = MiniBatchKMeans(n_clusters=dictionary_size, batch_size=100, verbose=1)
    dictionary.fit(np.array(desc_list))
    return dictionary

def extract_bow_features(descriptors, dictionary):
    features = np.zeros(dictionary.n_clusters)
    predictions = dictionary.predict(descriptors)
    for p in predictions:
        features[p] += 1
    return features

def build_dataset_sklearn(dir_names, file_paths, file_dir_indexes, dictionary, xfeatures2d):
    assert len(file_paths) == len(file_dir_indexes)    
    X = [] # data
    y = file_dir_indexes # labels

    print('Computing features using Bag-of-Words dictionary..')
    num_files = len(file_paths)
    bar = progressbar.ProgressBar(maxval=num_files).start()
    for i in range(num_files):
        p = file_paths[i]
        im = cv2.imread(p, 1)
        gray = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)
        kp, dsc = xfeatures2d.detectAndCompute(gray, None)
        features = extract_bow_features(dsc, dictionary)
        X.append(features)
        bar.update(i)
    bar.finish()
    return X, y