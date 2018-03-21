import argparse
import extractor as vgg16_extractor
import pickle
import numpy as np
import cv2
from helper import get_dir_info, build_dataset_vgg16, build_dataset_sklearn
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
	X_f = None
	results = []    

	# get test data info
	dir_names, file_paths, file_dir_indexes = get_dir_info(test_dir_path)

	# evaluate vgg16 extractor
	if vgg16_model is not None:
		model = pickle.load(open(vgg16_model, 'rb'))
		extractor = vgg16_extractor.build_extractor(vgg16_extractor.EXPECTED_DIM)

		for exp, values in experiments.iteritems():
			for v in values:
				if exp.lower() == 'rotate':
					X, y = build_dataset_vgg16(dir_names, file_paths, file_dir_indexes, extractor, expected_size=vgg16_extractor.EXPECTED_SIZE, rotation=v)
				elif exp.lower() == 'scale':
					X, y = build_dataset_vgg16(dir_names, file_paths, file_dir_indexes, extractor, expected_size=vgg16_extractor.EXPECTED_SIZE, scale=1 + v / 100.0)

				accuracy = model.score(X, y)
				results.append(("VGG16 " + type(model).__name__, exp, v, accuracy))


	# evaluate sift extractor
	if sift_dict is not None and sift_model is not None:
		sift = cv2.xfeatures2d.SIFT_create()
		dictionary = pickle.load(open(sift_dict, 'rb'))
		model = pickle.load(open(sift_model, 'rb'))

		for exp, values in experiments.iteritems():
			for v in values:
				if exp.lower() == 'rotate':
					X, y = build_dataset_sklearn(dir_names, file_paths, file_dir_indexes, dictionary, sift, rotation=v)
				elif exp.lower() == 'scale':
					X, y = build_dataset_sklearn(dir_names, file_paths, file_dir_indexes, dictionary, sift, scale=1 + v / 100.0)

				accuracy = model.score(X, y)
				results.append(("SIFT " + type(model).__name__, exp, v, accuracy))


	# evaluate surf extractor
	if surf_dict is not None and surf_model is not None:
		surf = cv2.xfeatures2d.SURF_create()
		dictionary = pickle.load(open(surf_dict, 'rb'))
		model = pickle.load(open(surf_model, 'rb'))
		for exp, values in experiments.iteritems():
			for v in values:
				if exp.lower() == 'rotate':
					X, y = build_dataset_sklearn(dir_names, file_paths, file_dir_indexes, dictionary, surf, rotation=v)
				elif exp.lower() == 'scale':
					X, y = build_dataset_sklearn(dir_names, file_paths, file_dir_indexes, dictionary, surf, scale=1 + v / 100.0)

				accuracy = model.score(X, y)
				results.append(("SURF " + type(model).__name__, exp, v, accuracy))


	print('\n\nResults:')
	for r in results:
		print('\t'.join([ str(v) for v in r ]))