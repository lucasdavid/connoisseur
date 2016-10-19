"""Transfer and fine-tune multiple architectures on van Gogh dataset.

Uses a network trained over `imagenet` and fine-tune it to van Gogh dataset.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import os

import numpy as np
import tensorflow as tf
from artificial.utils.experiments import ExperimentSet, Experiment, arg_parser
from connoisseur import datasets
from connoisseur.fusion import KerasFusion
from keras import applications
from keras import callbacks, optimizers, backend as K
from keras.engine import Input, Model
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization


class TrainingExperiment(Experiment):
    def setup(self):
        c = self.consts

        if c.ckpt_file:
            os.makedirs(os.path.dirname(c.ckpt_file), exist_ok=True)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        K.set_session(s)

        np.random.seed(c.seed)

    def run(self):
        c = self.consts

        if c.base_model in ('InceptionV3', 'Xception'):
            # These ones have a different pre-process function.
            from keras.applications.inception_v3 import preprocess_input
        else:
            from keras.applications.imagenet_utils import preprocess_input

        tf.logging.info('loading Van-Gogh data set...')
        van_gogh = datasets.VanGogh(c).download().extract().check().load()

        # Loading data...
        X, y = van_gogh.train_data
        X_test, y_test = van_gogh.test_data

        del van_gogh

        X = X.reshape((-1,) + X.shape[2:])
        y = np.repeat(y, c.train_n_patches)

        X, X_test = map(preprocess_input, (X, X_test))

        tf.logging.debug('training data shape: %s', X.shape)
        tf.logging.debug('validation data shape: %s', X_test.shape)

        with tf.device(c.device):
            tf.logging.info('host device: %s, model: %s',
                            c.device, c.base_model)

            images = Input(batch_shape=[None] + c.image_shape)
            base_model_cls = getattr(applications, c.base_model)
            base_model = base_model_cls(weights='imagenet',
                                        input_tensor=images,
                                        include_top=False)

            base_model.trainable = not c.frozen_base

            x = base_model.output
            x = Flatten(name='flatten')(x)
            x = Dense(4096, name='fc1')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(.5)(x)
            x = Dense(4096, name='fc2')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(.5)(x)
            x = Dense(2, name='predictions')(x)
            x = BatchNormalization()(x)
            x = Activation('softmax')(x)

            model = Model(input=base_model.input, output=x)

            if c.optimizer == 'sgd-optimized':
                opt = optimizers.SGD(lr=.001, momentum=0.9,
                                     decay=1e-6, nesterov=True)
            elif c.optimizer == 'adam-optimized':
                opt = optimizers.Adam(lr=.0001)
            else:
                opt = c.optimizer

            model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        try:
            tf.logging.info('training...')

            _calls = [
                callbacks.TensorBoard(c.tensorboard_file, write_graph=False),
                callbacks.EarlyStopping(patience=c.patience),
                callbacks.ModelCheckpoint(c.ckpt_file, save_best_only=True, verbose=1)
            ]
            model.fit(X, y, nb_epoch=c.n_epochs, validation_split=.2, batch_size=c.batch_size, callbacks=_calls)
        except KeyboardInterrupt:
            tf.logging.warning('training interrupted by user.')

        if c.ckpt_file:
            tf.logging.debug('loading weights from: %s' % c.ckpt_file)
            model.load_weights(c.ckpt_file)

        for strategy in ('farthest', 'sum', 'most_frequent'):
            tf.logging.info('score on testing data, using strategy `%s`: %.2f', strategy,
                            KerasFusion(model, strategy=strategy).score(X_test, y_test))

    def teardown(self):
        K.clear_session()


if __name__ == '__main__':
    args = arg_parser.parse_args()

    print(__doc__, flush=True)

    logging.basicConfig(level=logging.INFO, filename='./run.log')
    for logger in ('artificial', 'tensorflow'):
        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)

    (ExperimentSet(experiment_cls=TrainingExperiment)
     .load_from_json(args.constants)
     .run())
