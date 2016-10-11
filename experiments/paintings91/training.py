"""Paintings91 Connoisseur Training.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc
import logging
import os
from datetime import datetime

import tensorflow as tf
from connoisseur import Connoisseur, datasets, utils
from keras.applications import VGG19
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Flatten


class Paintings91(Connoisseur, metaclass=abc.ABCMeta):
    def build_model(self):
        consts = self.constants

        images = Input(batch_shape=[consts.batch_size] + consts.image_shape)
        base_model = VGG19(include_top=False, weights='imagenet',
                           input_tensor=images)

        n_classes = 91 if consts.classes is None else len(consts.classes)

        s = Flatten()(base_model.output)
        s = Dense(2048, activation='relu', name='fc1')(s)
        s = Dense(2048, activation='relu', name='fc2')(s)
        s = Dense(n_classes, activation='softmax', name='predictions')(s)

        model = Model(input=base_model.input, output=s)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def data(self, phase='training'):
        consts = self.constants

        with tf.device('/cpu'):
            dataset = datasets.Paintings91(consts.data_dir)
            data_generator = dataset.as_keras_generator()

            return data_generator.flow_from_directory(
                os.path.join(consts.data_dir, 'Images'),
                target_size=consts.image_shape[:2],
                classes=consts.classes,
                batch_size=consts.batch_size,
                seed=consts.seed)


def main():
    consts = utils.ExperimentSet('./training-constants.json')

    tf.logging.set_verbosity(tf.logging.INFO)
    log_filename = os.path.join(consts.logging_dir,
                                'training-' + str(datetime.now()) + '.log')
    logging.basicConfig(level=logging.INFO, filename=log_filename)

    c = Paintings91(constants=consts)

    try:
        t = utils.Timer()

        with tf.device(consts.device):
            data = c.data(phase='training')
            model = c.build_model()
            history = model.fit_generator(data, consts.n_samples_per_epoch,
                                          consts.n_iterations)
            print(history)

        tf.logging.info('done (%s)' % t)

    except KeyboardInterrupt:
        tf.logging.info('interrupted by the user')


if __name__ == '__main__':
    print(__doc__, flush=True)
    main()
