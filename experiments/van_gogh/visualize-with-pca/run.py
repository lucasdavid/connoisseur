# coding: utf-8
"""Visualize van Gogh dataset.

Uses VGG trained over `imagenet` to transform paintings in VanGogh dataset
into their low-dimensional representations and, finally, exploits PCA to
find the best 3-dimensional representation for the data.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import os
from math import ceil

import matplotlib
import numpy as np
import tensorflow as tf
from artificial.utils.experiments import ExperimentSet, Experiment, arg_parser
from keras import backend as K
from keras.applications import VGG19
from keras.engine import Input
from sklearn.decomposition import PCA

from connoisseur import datasets

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Axes3D


class VisualizingWithPCAExperiment(Experiment):
    def setup(self):
        c = self.consts
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        K.set_session(s)
        np.random.seed(c.seed)

    def run(self):
        c = self.consts
        van_gogh = datasets.VanGogh(c).download().extract().split_train_valid().as_keras_generator()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        K.set_session(s)

        with tf.device(c.device):
            tf.logging.info('host device: %s', c.device)

            # Build VGG model.
            images = Input(batch_shape=[None] + c.image_shape)
            model = VGG19(weights='imagenet', input_tensor=images,
                          include_top=False)

        train_data = van_gogh.flow_from_directory(
            directory=os.path.join(c.data_dir, 'vgdb_2016', 'train'),
            shuffle=False,
            target_size=c.image_shape[:2],
            classes=c.classes,
            batch_size=c.batch_size,
            seed=c.train_seed)

        tf.logging.info('extracting features...')
        n_batches = int(ceil(train_data.N / train_data.batch_size))

        X, y = [], []
        for batch in range(n_batches):
            # Load one batch of images.
            _X, _y = next(train_data)
            # Transform them using NN.
            _X = model.predict_on_batch(_X)
            # Flatten arrays.
            X.append(_X.reshape(_X.shape[0], -1))
            y.append(np.argmax(_y, axis=1))
        X, y = map(np.concatenate, (X, y))

        pca = PCA(n_components=3)
        Z = pca.fit_transform(X, y)

        tf.logging.info('%s --embedding--> %s', X.shape, Z.shape)
        tf.logging.info('components variance: %s (total: %f.2f)',
                        pca.explained_variance_ratio_,
                        np.sum(pca.explained_variance_ratio_))

        self.plot(Z, y)

    def plot(self, X, y):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        plt.grid()
        plt.savefig('embedding.png', bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    args = arg_parser.parse_args()

    print(__doc__, flush=True)

    logging.basicConfig(level=logging.INFO, filename='./run.log')
    for logger in ('artificial', 'tensorflow'):
        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)

    (ExperimentSet(experiment_cls=VisualizingWithPCAExperiment)
     .load_from_json(args.constants)
     .run_all())
