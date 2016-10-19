"""Histogram Colors and Predict on van Gogh dataset.

This experiment consists on the following procedures:

 * Extract features from paintings using InceptionV3 pre-trained over imagenet
 * Histogram color information over each painting and convert it to a feature vector
 * Concatenate the information
 * Reduce the dimensionality of the data with PCA
 * Train an SVM using the feature-vector obtained in (4)
 * Classify each patch of each test painting with an SVM machine
 * Fuse the predicted classes for each patch, resulting in the predicted class for the painting

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.applications import InceptionV3
from keras.engine import Input
from keras.engine import Model
from sacred import Experiment
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from connoisseur import datasets
from connoisseur.fusion import SkLearnFusion

ex = Experiment('histogram-colors-extract-cnn-train-svm')


@ex.config
def config():
    svm_seed = 2
    dataset_seed = 3
    classes = None
    batch_size = 32
    image_shape = [299, 299, 3]
    device = "/gpu:0"
    n_jobs = 8

    histogram_type = 'flattened'
    n_color_histogram_bins = 32

    load_mode = 'exact'
    train_n_patches = 40
    train_augmentations = []
    test_n_patches = 40
    test_augmentations = []
    data_dir = '/datasets/ldavid/van_gogh'

    grid_searching = False
    param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['auto', 0.01, 0.1, 1]}
    C = 1.0


def preprocess_input(x):
    x = x / 255.
    x -= 0.5
    x *= 2.
    return x


def compute_color_flattened_histogram(x, n_color_histogram_bins):
    n_samples, n_patches, height, width, channels = x.shape

    x = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
    x = np.array([np.array([cv2.calcHist([x[:, :, c]], [0], None, [n_color_histogram_bins], [0, 256])
                            for c in range(channels)]).ravel() for x in x])
    x = x.reshape(n_samples, n_patches, n_color_histogram_bins * channels)
    return x


def compute_multidimensional_histogram(x, n_color_histogram_bins):
    n_samples, n_patches, height, width, channels = x.shape

    x = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
    x = np.array([cv2.calcHist([x], list(range(channels)), None, channels * [n_color_histogram_bins],
                               channels * [0, 256]).ravel() for x in x])
    x = x.reshape(n_samples, n_patches, n_color_histogram_bins ** channels)
    return x


@ex.automain
def run(dataset_seed, svm_seed, image_shape, batch_size, data_dir, load_mode, train_n_patches,
        train_augmentations, test_n_patches, test_augmentations, device, grid_searching, param_grid, n_jobs,
        histogram_type, n_color_histogram_bins,
        C):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    K.set_session(s)

    vangogh = datasets.VanGogh(
        base_dir=data_dir, image_shape=image_shape,
        load_mode=load_mode,
        train_n_patches=train_n_patches,
        test_n_patches=test_n_patches,
        train_augmentations=train_augmentations,
        test_augmentations=test_augmentations,
        random_state=dataset_seed
    ).download().extract().check()

    with tf.device(device):
        images = Input(batch_shape=[None] + image_shape)
        base_model = InceptionV3(weights='imagenet', input_tensor=images, include_top=False)
        x = base_model.output
        x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = layers.Flatten(name='flatten')(x)
        fc_model = Model(input=base_model.input, output=x)

    print('embedding data...')
    X, y = vangogh.load('train').get('train')
    vangogh.unload('train')
    Z = fc_model.predict(preprocess_input(X).reshape((-1,) + X.shape[2:])).reshape(X.shape[:2] + (-1,))
    X_test, y_test = vangogh.load('test').get('test')
    vangogh.unload('test')
    Z_test = fc_model.predict(preprocess_input(X_test).reshape((-1,) + X_test.shape[2:])).reshape(
        X_test.shape[:2] + (-1,))
    del fc_model, base_model
    K.clear_session()

    print('negative samples:', (y == 0).sum())
    print('positive samples:', (y == 1).sum())

    print('train embedding shape and size: %s, %f MB' % (Z.shape, Z.nbytes / 1024 ** 2))
    print('test embedding shape and size: %s, %f MB' % (Z_test.shape, Z_test.nbytes / 1024 ** 2))

    if histogram_type == 'flattened':
        print('computing flattened color histograms for train paintings...')
        X = compute_color_flattened_histogram(X, n_color_histogram_bins)
        print('computing multidimensional color histograms for test paintings...')
        X_test = compute_color_flattened_histogram(X_test, n_color_histogram_bins)
    elif histogram_type == 'multidimensional':
        print('computing multidimensional color histograms for train paintings...')
        X = compute_multidimensional_histogram(X, n_color_histogram_bins)
        print('computing multidimensional color histograms for test paintings...')
        X_test = compute_multidimensional_histogram(X_test, n_color_histogram_bins)

    print('train histogram shape and size: %s, %f MB' % (X.shape, X.nbytes / 1024 ** 2))
    print('test histogram shape and size: %s, %f MB' % (X_test.shape, X_test.nbytes / 1024 ** 2))

    histogram_pca = PCA(n_components=.99)
    n_samples, n_patches, n_features = X.shape
    X = histogram_pca.fit_transform(X.reshape(-1, n_features)).reshape(n_samples, n_patches, -1)
    n_samples, n_patches, n_features = X_test.shape
    X_test = histogram_pca.transform(X_test.reshape(-1, n_features)).reshape(n_samples, n_patches, -1)
    print('dimensionality reduction on histogram signal: R^%i->R^%i (%.2f variance kept)'
          % (n_features, histogram_pca.n_components_, np.sum(histogram_pca.explained_variance_ratio_)))
    print('train histogram shape and size: %s, %f MB' % (X.shape, X.nbytes / 1024 ** 2))
    print('test histogram shape and size: %s, %f MB' % (X_test.shape, X_test.nbytes / 1024 ** 2))

    # Concatenate CNN features with color features.
    X, X_test = np.concatenate((Z, X), axis=2), np.concatenate((Z_test, X_test), axis=2)
    del Z, Z_test
    X, y = X.reshape(-1, X.shape[-1]), np.repeat(y, train_n_patches)

    print('train feature-vector shape and size: %s, %f MB' % (X.shape, X.nbytes / 1024 ** 2))
    print('test feature-vector shape and size: %s, %f MB' % (X_test.shape, X_test.nbytes / 1024 ** 2))

    if grid_searching:
        print('grid searching...')
        flow = Pipeline([
            ('pca', PCA(n_components=.999)),
            ('svc', LinearSVC(dual=X.shape[0] <= X.shape[1]))
        ])
        grid = GridSearchCV(estimator=flow, param_grid=param_grid, n_jobs=n_jobs, verbose=1)
        grid.fit(X, y)
        print('best params: %s' % grid.best_params_)
        print('best score on validation data: %.2f' % grid.best_score_)
        model = grid.best_estimator_
    else:
        print('training...')
        # These are the best parameters I've found so far.
        model = Pipeline([
            ('pca', PCA(n_components=.999)),
            ('svc', LinearSVC(C=C, dual=X.shape[0] <= X.shape[1], class_weight='balanced'))
        ])
        model.fit(X, y)

    pca = model.named_steps['pca']
    print('dimensionality reduction: R^%i->R^%i (%.2f variance kept)'
          % (X.shape[-1], pca.n_components_, np.sum(pca.explained_variance_ratio_)))
    print('training classification report:')
    p = model.predict(X)
    print(metrics.classification_report(y, p))
    print('train score: %.2f' % metrics.accuracy_score(y, p), '\n',
          metrics.classification_report(y, p), '\nConfusion matrix:\n',
          metrics.confusion_matrix(y, p))
    del X, y, p

    for strategy in ('farthest', 'sum', 'most_frequent'):
        f = SkLearnFusion(model, strategy=strategy)
        p_test = f.predict(X_test)
        print('score using', strategy, 'strategy: %.2f' % metrics.accuracy_score(y_test, p_test), '\n',
              metrics.classification_report(y_test, p_test), '\nConfusion matrix:\n',
              metrics.confusion_matrix(y_test, p_test))
