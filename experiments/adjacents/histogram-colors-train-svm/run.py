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
import json
import os

import matplotlib

matplotlib.use('Agg')

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from sacred import Experiment
from sklearn.preprocessing import LabelEncoder

from connoisseur.utils.image import load_img, img_to_array

ex = Experiment('7-histogram-colors-train-svm')


@ex.config
def config():
    n_bins = 64
    histogram_variability = .95
    histogram_range = (0, 255)

    train_dir = '/work/ldavid/datasets/vangogh/vgdb_2016/extracted_patches/train'
    valid_dir = '/work/ldavid/datasets/vangogh/vgdb_2016/extracted_patches/valid'
    test_dir = '/work/ldavid/datasets/vangogh/vgdb_2016/extracted_patches/test'

    svm_seed = 2
    n_jobs = 8
    grid_searching = False
    param_grid = {'C': [.01, .1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    C = 1.0
    maintain_energy = 1.0


def load_data(directory):
    X, y, names = [], [], []
    labels = os.listdir(directory)

    if not labels:
        raise ValueError('No labels detected. Perhaps the pointed directory is wrong: %s'
                         % directory)

    for label in labels:
        samples = os.listdir(os.path.join(directory, label))

        for sample in samples:
            x = img_to_array(load_img(os.path.join(directory, label, sample), mode='hsv'))
            X.append(x)
            y.append(label)
            names.append(sample)

    X = np.array(X, copy=False)
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y, names, le


def compute_label_histograms(X, y, le, n_bins, histogram_range):
    histograms = []

    for channel_id, channel in enumerate('HSV'):
        print('processing', channel, end=' ')

        handles = []

        for c in le.classes_:
            c_ = le.transform([c])[0]
            X_c = X[y == c_]

            hist, bins = np.histogram(X_c[:, :, :, channel_id].flatten(),
                                      bins=n_bins, range=histogram_range)
            hist = hist / X_c.shape[0]

            handler, = plt.plot(bins[:-1], hist, label=c)
            handles.append(handler)
            histograms.append(hist)

        plt.legend(handles=handles)
        plt.savefig('./results/%s.png' % channel)
        plt.clf()

        print('(done)')
    return histograms


def compute_data_histograms(X, n_bins, histogram_range):
    channel_id = 2  # Value channel (HS*V*)
    return np.array(list(map(lambda _x: _x[0],
                             [np.histogram(x[:, :, channel_id].flatten(), bins=n_bins, range=histogram_range)
                              for x in X])), copy=False)


@ex.automain
def run(train_dir, valid_dir, test_dir,
        n_bins, histogram_variability, histogram_range,
        svm_seed, maintain_energy, grid_searching, param_grid, n_jobs, C):
    os.makedirs('./results', exist_ok=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    K.set_session(s)

    X, y, names, le = load_data(directory=train_dir)
    # compute_label_histograms(X, y, le, n_bins, histogram_range)
    X = compute_data_histograms(X, n_bins, histogram_range)

    X_valid, y_valid, names_valid, _ = load_data(directory=valid_dir)
    X_valid = compute_data_histograms(X_valid, n_bins, histogram_range)

    X_test, y_test, names_test, _ = load_data(directory=test_dir)
    X_test = compute_data_histograms(X_test, n_bins, histogram_range)

    print(*('%s histogram shape and size: %s, %f MB' % (phase, _X.shape, _X.nbytes / 1024 ** 2)
            for phase, _X in (('train', X), ('valid', X_valid), ('test', X_test))),
          sep='\n')

    X_features = X.shape[1]

    if maintain_energy < 1.0:
        histogram_pca = PCA(n_components=histogram_variability)
        X = histogram_pca.fit_transform(X)
        X_valid = histogram_pca.transform(X_valid)
        X_test = histogram_pca.transform(X_test)

        print('dimensionality reduction on histogram signal: R^%i->R^%i (%.2f variance kept)'
              % (X_features, histogram_pca.n_components_, np.sum(histogram_pca.explained_variance_ratio_)))
        print(*('reduced %s histogram shape and size: %s, %f MB' % (phase, _X.shape, _X.nbytes / 1024 ** 2)
                for phase, _X in (('train', X), ('valid', X_valid), ('test', X_test))),
              sep='\n')
        del histogram_pca

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

    results = []
    for phase, _X, _y, _names in (('train', X, y, names),
                                  ('valid', X_valid, y_valid, names_valid),
                                  ('test', X_test, y_test, names_test)):
        p = model.predict(_X)
        accuracy_score = model.score(_X, _y)
        print('score on %s data: %f' % (phase, accuracy_score))

        results.append({
            'phase': phase,
            'samples': _names,
            'p': p.tolist()
        })

    with open('./predictions.json', 'w') as f:
        json.dump(results, f)
