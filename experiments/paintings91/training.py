"""Paintings91 Connoisseur Training.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc
import os

import tensorflow as tf
from connoisseur import Connoisseur, datasets
from connoisseur.utils import Timer
from keras.applications import VGG19
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Flatten

BASE_DIR = '/media/ldavid/hdd/research'
LOGGING_DIR = os.path.join(BASE_DIR, 'logs', 'paintings91')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'Paintings91')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'paintings91.h5')

N_CLASSES = None

N_EPOCHS = 100
BATCH_SIZE = 4
SAMPLES_PER_EPOCH = 4096
SEED = None

IMAGE_SHAPE = (224, 224, 3)


class Paintings91(Connoisseur, metaclass=abc.ABCMeta):
    def build(self):
        images = Input(batch_shape=(BATCH_SIZE,) + IMAGE_SHAPE)
        base_model = VGG19(include_top=False, weights='imagenet',
                           input_tensor=images)

        n_classes = 91 if N_CLASSES is None else len(N_CLASSES)

        s = Flatten()(base_model.output)
        s = Dense(1024, activation='relu', name='fc1')(s)
        s = Dense(1024, activation='relu', name='fc2')(s)
        s = Dense(n_classes, activation='softmax', name='predictions')(s)

        model = Model(input=base_model.input, output=s)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def data(self, phase='training'):
        with tf.device('/cpu'):
            dataset = datasets.Paintings91(DATA_DIR)
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
        t = Timer()

        with tf.device('/gpu:0'):
            data = c.data(phase='training')
            model = c.build()
            history = model.fit_generator(data, SAMPLES_PER_EPOCH, N_EPOCHS)
            print(history)

        tf.logging.info('done (%s)' % t)

    except KeyboardInterrupt:
        tf.logging.info('interrupted by the user')


if __name__ == '__main__':
    print(__doc__, flush=True)
    main()
