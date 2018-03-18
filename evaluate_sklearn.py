import sys
import tables
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# config
classfiers = [
	LogisticRegression(),
	SVC(),    
	MLPClassifier(),
	DecisionTreeClassifier(),
	GradientBoostingClassifier(),
	RandomForestClassifier(),
]
CV = 7

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluate scikit-learn classifiers using extracted dataset features', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('train_file', help="Path to train data (features) input file")
	parser.add_argument('test_file', help="Path to test data (features) input file")
	parser.add_argument('--n_folds', type=int, default=CV, help="Number of folds (K) for K-fold cross validation")

	args = parser.parse_args()
	train_file = args.train_file
	test_file = args.test_file
	n_folds = args.n_folds

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
	y = np.concatenate((train_dataset.labels[:].argmax(1), test_dataset.labels[:].argmax(1)), axis=0)

	# close dataset
	train_datafile.close()
	test_datafile.close()

	print('Cross validation with k={}..'.format(CV))
	for classifier in classfiers:
		# cross_validate
		scores = cross_val_score(classifier, X, y, cv=CV)
		print("{} CV accuracy: {:0.2f} (+/- {:0.2f})".format(type(classifier).__name__, scores.mean(), scores.std() * 2))
