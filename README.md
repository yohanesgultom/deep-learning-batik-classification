# Deep learning batik classification

Indonesian Batik classification using **VG16 convolutional neural network (CNN)** as feature extractor + softmax as classifier.

Dataset consist of 5 batik classes where each images will **belong to exactly one class**:

1. Parang: parang (traditional blade) motifs
1. Lereng: also blade-like pattern but less pointy than Parang
1. Ceplok: repetition of geometrical motif shapes (eg. rectangle, square, ovals, stars)
1. Kawung: kawung fruits motif
1. Nitik: flowers-like motifs

## Requirements

To run the experiments, you would need:

* CUDA device with global RAM >= 4 GB (tested with GTX 980)
* Python 2.7.x
* Virtualenv (optional. For isolated environment)

> The programs can also run on CPU. You just need to use `tensorflow` instead of `tensorflow-gpu`. Further info https://www.tensorflow.org/install/

The programs expect batik image files (jpg/png) to be grouped by their classes. In this experiment, below directory structures are used:

```
train_data_dir/
	Ceplok/*.jpg
	Kawung/*.jpg
	Lereng/*.jpg
	Nitik/*.jpg
	Parang/*.jpg

test_data_dir/
	Ceplok/*.jpg
	Kawung/*.jpg
	Lereng/*.jpg
	Nitik/*.jpg
	Parang/*.jpg
```

## Installation

Install dependencies: `pip install -r requirements.txt`

## Experiments

### VGG16 + Softmax NN

Following commands are executing these steps:

1. Convert images dataset to vector [h5 format](http://www.h5py.org/):  
2. Extract features from dataset (still in h5 format)
3. Train & evaluate 4 variations of NN (vary by number of hidden layers and activation function) with cross validation

```
python preprocess.py train_data_dir train.h5
python preprocess.py test_data_dir test.h5
python extractor.py train.h5 train.features.h5
python extractor.py test.h5 test.features.h5
python train.py train.features.h5 test.features.h5
```
> Process time is around 40 minutes with GTX 980 4 GB

### Classification using SIFT Bag of Words + SVM

Following command is executing these steps:

1. Extract SIFT descriptors from images
2. Cluster descriptors to build vocabulary using K-means
3. Extract bag of words features from images using vocabulary
4. Train & evaluate 6 classifiers with cross validation

```
python siftbow.py train_data_dir test_data_dir
```

> Process time is around 52 minutes with Intel Core i7 5500U 8 GB RAM

### Classification using SURF Bag of Words + SVM

Following command is executing these steps:

1. Extract SURF descriptors from images
2. Cluster descriptors to build vocabulary using K-means
3. Extract bag of words features from images using vocabulary
4. Train & evaluate 6 classifiers with cross validation

```
python surfbow.py train_data_dir test_data_dir
```

> Process time is around 45 minutes with Intel Core i7 5500U 8 GB RAM