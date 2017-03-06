"""Histogram Colors and Predict on van Gogh dataset.

This experiment consists on the following procedures:

 * Extract features from paintings using InceptionV3 pre-trained over imagenet
 * Histogram color information over each painting and convert it to a feature vector
 * Concatenate the information
 * Reduce the dimensionality of the data with PCA
 * Train an SVM using the feature-vector obtained in (4)
 * Classify each patch of each test painting with an SVM machine
 * Fuse the predicted classes for each patch, resulting in the predicted class for the painting

Author: Lucas David -- <lucasolivdavid@gmail.com>
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

ex = Experiment('inception-pca-histogram-pca-svm')


@ex.config
def config():
    svm_seed = 2
    dataset_seed = 3
    classes = None
    batch_size = 32
    image_shape = [299, 299, 3]
    device = "/gpu:1"
    n_jobs = 8

    histogram_type = 'flattened'
    n_color_histogram_bins = 128
    histogram_variability = .9

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
def run(_run, dataset_seed, svm_seed, image_shape, batch_size, data_dir, load_mode, train_n_patches,
        train_augmentations, test_n_patches, test_augmentations, device, grid_searching, param_grid, n_jobs,
        histogram_type, n_color_histogram_bins, histogram_variability,
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
    ).download().extract().split()

    with tf.device(device):
        images = Input(batch_shape=[None] + image_shape)
        base_model = InceptionV3(weights='imagenet', input_tensor=images, include_top=False)
        x = base_model.output
        x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = layers.Flatten(name='flatten')(x)
        fc_model = Model(input=base_model.input, output=x)

    print('embedding data...')
    X, y = vangogh.load_patches_from_full_images('train').get('train')
    vangogh.unload('train')
    Z = fc_model.predict(preprocess_input(X).reshape((-1,) + X.shape[2:])).reshape(X.shape[:2] + (-1,))
    X_test, y_test = vangogh.load_patches_from_full_images('test').get('test')
    vangogh.unload('test')
    Z_test = fc_model.predict(preprocess_input(X_test).reshape((-1,) + X_test.shape[2:])).reshape(
        X_test.shape[:2] + (-1,))
    del fc_model, base_model
    K.clear_session()

    inception_pca = PCA(n_components=.99)
    n_samples, n_patches, n_features = Z.shape
    Z = inception_pca.fit_transform(Z.reshape(-1, n_features)).reshape(n_samples, n_patches, -1)
    n_samples, n_patches, n_features = Z_test.shape
    Z_test = inception_pca.transform(Z_test.reshape(-1, n_features)).reshape(n_samples, n_patches, -1)
    print('dimensionality reduction on inception signal: R^%i->R^%i (%.2f variance kept)'
          % (n_features, inception_pca.n_components_, np.sum(inception_pca.explained_variance_ratio_)))
    print('train embedding shape and size: %s, %f MB' % (Z.shape, Z.nbytes / 1024 ** 2))
    print('test embedding shape and size: %s, %f MB' % (Z_test.shape, Z_test.nbytes / 1024 ** 2))
    del inception_pca

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

    histogram_pca = PCA(n_components=histogram_variability)
    n_samples, n_patches, n_features = X.shape
    X = histogram_pca.fit_transform(X.reshape(-1, n_features)).reshape(n_samples, n_patches, -1)
    n_samples, n_patches, n_features = X_test.shape
    X_test = histogram_pca.transform(X_test.reshape(-1, n_features)).reshape(n_samples, n_patches, -1)
    print('dimensionality reduction on histogram signal: R^%i->R^%i (%.2f variance kept)'
          % (n_features, histogram_pca.n_components_, np.sum(histogram_pca.explained_variance_ratio_)))
    print('train histogram shape and size: %s, %f MB' % (X.shape, X.nbytes / 1024 ** 2))
    print('test histogram shape and size: %s, %f MB' % (X_test.shape, X_test.nbytes / 1024 ** 2))
    del histogram_pca

    # Concatenate CNN features with color features.
    X, X_test = np.concatenate((Z, X), axis=2), np.concatenate((Z_test, X_test), axis=2)
    del Z, Z_test
    X, y = X.reshape(-1, X.shape[-1]), np.repeat(y, train_n_patches)

    print('train feature-vector shape and size: %s, %f MB' % (X.shape, X.nbytes / 1024 ** 2))
    print('test feature-vector shape and size: %s, %f MB' % (X_test.shape, X_test.nbytes / 1024 ** 2))

    if grid_searching:
        print('grid searching...')
        model = SVC(class_weight='balanced', random_state=svm_seed)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=n_jobs, verbose=1)
        grid.fit(X, y)
        print('best params: %s' % grid.best_params_)
        print('best score on validation data: %.2f' % grid.best_score_)
        model = grid.best_estimator_
    else:
        print('training...')
        # These are the best parameters I've found so far.
        model = SVC(C=C, class_weight='balanced', random_state=svm_seed)
        model.fit(X, y)

    accuracy_score = model.score(X, y)
    print('score on training data: %.2f' % accuracy_score)
    _run.info['accuracy'] = {
        'train': accuracy_score,
        'test': -1
    }
    del X, y

    print('testing...')
    for strategy in ('farthest', 'sum', 'most_frequent'):
        f = SkLearnFusion(model, strategy=strategy)
        p_test = f.predict(X_test)
        accuracy_score = metrics.accuracy_score(y_test, p_test)
        print('score using', strategy, 'strategy: %.2f' % accuracy_score, '\n',
              metrics.classification_report(y_test, p_test), '\nConfusion matrix:\n',
              metrics.confusion_matrix(y_test, p_test))

        if accuracy_score > _run.info['accuracy']['test']:
            _run.info['accuracy']['test'] = accuracy_score
