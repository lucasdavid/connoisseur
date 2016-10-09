"""Paintings91 Connoisseur Training.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from connoisseur import Connoisseur, datasets, utils
from keras.applications import VGG19
from keras.engine import Input
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

SEED = None
N_CLASSES = None
BATCH_SIZE = 18
N_ITERATIONS = 237
IMAGE_SHAPE = (224, 224, 3)
BASE_NETWORK = VGG19

BASE_DIR = '/media/ldavid/hdd/research'
LOGGING_FILE = os.path.join(BASE_DIR, 'logs', 'paintings91',
                            str(datetime.now()) + '.log')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'paintings91.h5')

DATA_DIR = '/media/ldavid/hdd/research/data/Paintings91'

logging.basicConfig(filename=LOGGING_FILE, level=logging.DEBUG)
logger = logging.getLogger('tensorflow')


class Paintings91(Connoisseur, metaclass=abc.ABCMeta):
    def build(self):
        images = Input(batch_shape=(BATCH_SIZE,) + IMAGE_SHAPE)
        return BASE_NETWORK(weights='imagenet',
                            input_tensor=images,
                            include_top=False)

    def data(self, phase='training'):
        with tf.device('/cpu'):
            dataset = datasets.Paintings91(DATA_DIR).check()
            data_generator = dataset.as_keras_generator()

            return data_generator.flow_from_directory(
                os.path.join(DATA_DIR, 'Images'),
                target_size=IMAGE_SHAPE[:2],
                classes=N_CLASSES,
                batch_size=BATCH_SIZE,
                seed=SEED)


def main():
    c = Paintings91()

    try:
        t = utils.Timer()
        data = c.data()
        X, y = [], []

        with tf.device('/gpu:0'):
            model = c.build()

            for i in range(N_ITERATIONS):
                _X, _y = next(data)
                X += [model.predict_on_batch(_X).reshape((BATCH_SIZE, -1))]
                y += [np.argmax(_y, axis=1)]

        X, y = np.concatenate(X), np.concatenate(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
        clf = LinearSVC()
        clf.fit(X_train, y_train)

        tf.logging.info('training score: %.2f%%', clf.score(X_train, y_train))
        tf.logging.info('Test score: %.2f%%', clf.score(X_test, y_test))
        tf.logging.info('done (%s)' % t)

    except KeyboardInterrupt:
        tf.logging.info('interrupted by the user')


if __name__ == '__main__':
    print(__doc__, flush=True)
    main()
