"""Computing and Showing Histograms.

Uses openCV to compute color histograms and over paintings in VanGogh
dataset and save those as JPG images.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from connoisseur import datasets
from sacred import Experiment

ex = Experiment('computing-and-showing-histograms')


@ex.config
def config():
    dataset_seed = 4
    image_shape = [299, 299, 3]
    data_dir = '/datasets/ldavid/van_gogh'
    n_color_histogram_bins = 256
    train_n_patches = 40
    train_augmentations = []
    test_n_patches = 40
    test_augmentations = []


def preprocess_input(x):
    x = x / 255.
    x -= 0.5
    x *= 2.
    return x


def compute_color_flattened_histogram(x, n_color_histogram_bins):
    n_samples, n_patches, height, width, channels = x.shape

    x = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
    x = np.array([np.array([cv2.calcHist([x[:, :, c]], [0], None, [n_color_histogram_bins], [0, 256])
                            for c in range(channels)]) for x in x])
    x = x.reshape(n_samples, n_patches, *x.shape[1:])
    return x


def compute_multidimensional_histogram(x, n_color_histogram_bins):
    n_samples, n_patches, height, width, channels = x.shape

    x = x.reshape((-1,) + x.shape[2:]).astype(np.float32)
    x = np.array([cv2.calcHist([x], list(range(channels)), None, channels * [n_color_histogram_bins],
                               channels * [0, 256]).ravel() for x in x])
    x = x.reshape(n_samples, n_patches, n_color_histogram_bins ** channels)
    return x


@ex.automain
def run(dataset_seed, image_shape, data_dir, n_color_histogram_bins,
        train_n_patches, train_augmentations,
        test_n_patches, test_augmentations):
    os.makedirs(data_dir, exist_ok=True)

    print('loading Van-Gogh data set...')
    vangogh = datasets.VanGogh(
        base_dir=data_dir, image_shape=image_shape,
        train_n_patches=train_n_patches,
        test_n_patches=test_n_patches,
        train_augmentations=train_augmentations,
        test_augmentations=test_augmentations,
        random_state=dataset_seed
    ).download().extract().split()

    for phase in ('train', 'test'):
        os.makedirs('./histograms/%s/' % phase, exist_ok=True)

        X, y = vangogh.load_patches_from_full_images(phase).get(phase)
        vangogh.unload(phase)
        print('%s data shape and size: %s, %f MB' % (phase, X.shape, X.nbytes / 1024 ** 2))

        print('computing color histograms for %s paintings...' % phase)
        X = compute_color_flattened_histogram(X, n_color_histogram_bins)
        print('%s histogram shape and size: %s, %f MB' % (phase, X.shape, X.nbytes / 1024 ** 2))

        for i, x in enumerate(X):
            os.makedirs('./histograms/%s/%i' % (phase, i))
            for j, patch in enumerate(x):
                colors = ('r', 'g', 'b')

                plt.figure()
                for h, c in zip(patch, colors):
                    plt.plot(h, color=c)
                    plt.xlim([0, 256])
                plt.savefig('./histograms/%s/%i/%i.jpg' % (phase, i, j))
