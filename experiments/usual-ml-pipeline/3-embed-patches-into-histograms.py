"""Embed Painting Patches into Histograms.

This experiment consists on the following procedures:

 * Extract features from paintings using InceptionV3 pre-trained over imagenet
 * Histogram color information over each painting and convert it to a feature vector
 * Save to disk

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
import pickle

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from sacred import Experiment
from sklearn.preprocessing import LabelEncoder

from connoisseur.utils.image import load_img, img_to_array

ex = Experiment('3-embed-patches-into-histograms')


@ex.config
def config():
    data_dir = '/path/to/patches'
    n_bins = 256
    histogram_range = (0, 255)


def load_data(directory):
    X, y, names = [], [], []
    labels = sorted(os.listdir(directory))

    if not labels:
        raise ValueError('No labels detected. Perhaps the pointed directory is wrong: %s'
                         % directory)

    for label in labels:
        samples = os.listdir(os.path.join(directory, label))

        X += [img_to_array(load_img(os.path.join(directory, label, sample), mode='hsv'))
              for sample in samples]
        y += len(samples) * [label]
        names += samples

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
def run(data_dir, n_bins, histogram_range):
    os.makedirs('./results', exist_ok=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    K.set_session(s)

    # X, y, names, le = load_data(directory=train_dir)
    # compute_label_histograms(X, y, le, n_bins, histogram_range)

    for phase in ('train', 'valid', 'test'):
        directory = os.path.join(data_dir, phase)

        if not os.path.exists(directory):
            print('%s data not present - skipping' % phase)
            continue

        X, y, names, le = load_data(directory=directory)
        X = compute_data_histograms(X, n_bins, histogram_range)

        print('%s histogram shape and size: %s, %f MB' % (phase, X.shape, X.nbytes / 1024 ** 2))

        with open('./results/%s.0.pickle' % phase, 'wb') as f:
            pickle.dump({'data': X, 'target': y, 'names': np.array(names, copy=False)},
                        f, pickle.HIGHEST_PROTOCOL)
