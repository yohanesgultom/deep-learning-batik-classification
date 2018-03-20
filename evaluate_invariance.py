import argparse
import extractor as vgg16_extractor
import evaluate_sklearn
import pickle
import numpy as np
from helper import read_and_transform_dataset, extract_bow_features
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotate and zoom images and test them with VGG16, SIFT and SURF classifiers', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('test_dir_path', help="Path to raw test dataset directory")
    parser.add_argument('--vgg16_model', help="VGG16 classifier .pkl")
    parser.add_argument('--sift_dict', help="SIFT BoW dictionary .pkl")
    parser.add_argument('--sift_model', help="SIFT scikit-learn classifier .pkl")
    parser.add_argument('--surf_dict', help="SURF BoW dictionary .pkl")
    parser.add_argument('--surf_model', help="SURF scikit-learn classifier .pkl")

    args = parser.parse_args()
    test_dir_path = args.test_dir_path
    vgg16_model = args.vgg16_model	
    sift_dict = args.sift_dict	
    sift_model = args.sift_model	
    surf_dict = args.surf_dict	
    surf_model = args.surf_model	

    experiments = {
        'rotate': [90, 180, 270],
        'scale': [10, 30, 50],
    }

    X = None
    y = None
    results = []
    for exp, values in experiments.iteritems():
        for v in values:
            # transform dataset
            if exp.lower() == 'rotate':
                X, y = read_and_transform_dataset(test_dir_path, rotate=v)
            elif exp.lower() == 'scale':
                X, y = read_and_transform_dataset(test_dir_path, scale=1 + v / 100.0)

            if vgg16_model is not None:
                model = pickle.load(open(vgg16_model, 'rb'))
                extractor = vgg16_extractor.build_extractor(vgg16_extractor.EXPECTED_DIM)
                X_f = extractor.predict(X, verbose=0)
                accuracy = model.score(X_f, y)
                results.append(('VGG16 + MLP', exp, v, accuracy))

            if sift_dict is not None and sift_model is not None:
                sift = cv2.xfeatures2d.SIFT_create()
                dictionary = pickle.load(open(sift_dict, 'rb'))
                model = pickle.load(open(sift_model, 'rb'))
                X_f = []
                for x in X:
                    kp, dsc = sift.detectAndCompute(x, None)
                    features = extract_bow_features(dsc, dictionary)
                    X_f.append(features)
                X_f = np.array(X_f)
                accuracy = model.score(X_f, y)
                results.append(('SIFT + MLP', exp, v, accuracy))

            if surf_dict is not None and surf_model is not None:
                surf = cv2.xfeatures2d.SURF_create()
                dictionary = pickle.load(open(surf_dict, 'rb'))
                model = pickle.load(open(surf_model, 'rb'))
                X_f = []
                for x in X:
                    kp, dsc = surf.detectAndCompute(x, None)
                    features = extract_bow_features(dsc, dictionary)
                    X_f.append(features)
                X_f = np.array(X_f)
                accuracy = model.score(X_f, y)
                results.append(('SURF + MLP', exp, v, accuracy))
    
    print('\n\nResults:')
    for r in results:
        print('\t'.join([ str(v) for v in r ]))