"""VanGogh Connoisseur Straight Predicting.

Uses VGG trained over `imagenet` to transform paintings in VanGogh dataset
into their low-dimensional representations and, finally, exploits LinearSVM
to classify these paintings.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import abc
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from connoisseur import Connoisseur, datasets
from connoisseur.utils import ExperimentSet, Experiment, arg_parser
from keras import backend as K
from keras.applications import VGG19
from keras.engine import Input
from keras.engine import Model
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class VanGogh(Connoisseur, metaclass=abc.ABCMeta):
    def build_model(self):
        consts = self.constants
        images = Input(batch_shape=[consts.batch_size] + consts.image_shape)
        base_model = VGG19(weights='imagenet', input_tensor=images)

        self.model_ = Model(input=base_model.input,
                            output=base_model.get_layer('fc2').output)
        return self.model_

    def build_dataset(self):
        consts = self.constants

        with tf.device('/cpu'):
            dataset = datasets.VanGogh(consts.data_dir)
            g = dataset.download().extract().check().as_keras_generator()

        self.dataset_ = g
        return self.dataset_

    def data(self, phase='train'):
        assert self.dataset_, ('Dataset not built yet. Did you forget to '
                               'call .build_dataset()?')
        assert phase in ('train', 'test')

        consts = self.constants
        folder = os.path.join(consts.data_dir, 'vgdb_2016', phase)

        seed = getattr(consts, '%s_seed' % phase)

        with tf.device('/cpu'):
            print('batch size: ', consts.batch_size)
            return self.dataset_.flow_from_directory(
                folder,
                target_size=consts.image_shape[:2],
                classes=consts.classes,
                batch_size=consts.batch_size,
                seed=seed)


class StraightPredictionExperiment(Experiment):
    def run(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        K.set_session(s)

        consts = self.consts

        os.makedirs(consts.logging_dir, exist_ok=True)
        os.makedirs(consts.models_dir, exist_ok=True)

        tf.logging.set_verbosity(tf.logging.INFO)
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(consts.logging_dir,
                                                  str(datetime.now()) + '.log'))

        with tf.device(consts.device):
            c = VanGogh(constants=consts, build=True)

            tf.logging.info('host device: %s', consts.device)

            tf.logging.info('training...')
            X, y = self.extract_features(c, c.data('train'))

            grid = GridSearchCV(SVC(),
                                consts.grid_search_params,
                                n_jobs=consts.n_jobs,
                                cv=None,
                                verbose=1)
            grid.fit(X, y)

            tf.logging.info('best params: %s', grid.best_params_)
            tf.logging.info('best score on validation data: %.2f',
                            grid.best_score_)
            self.log_score(grid, X, y)

            tf.logging.info('testing...')
            X, y = self.extract_features(c, c.data('test'))
            test_score = self.log_score(grid, X, y)

            joblib.dump(grid, os.path.join(
                consts.models_dir,
                'model-%.2f-%.2f.pkl' % (grid.best_score_, test_score)))

            del grid, c, X, y

        del s

    def extract_features(self, c, data):
        tf.logging.info('extracting features...')

        X, y = [], []

        for _ in range(data.N // data.batch_size):
            # Transform images into their low-dimensional representations.
            _X, _y = next(data)
            _X = c.model_.predict_on_batch(_X)
            _X = _X.reshape((self.consts.batch_size, -1))

            X.append(_X)
            y.append(np.argmax(_y, axis=1))

        X, y = np.concatenate(X), np.concatenate(y)

        tf.logging.info('done (X.shape: %s)', X.shape)

        return X, y

    def log_score(self, clf, X, y):
        consts = self.consts

        overall_score = clf.score(X, y)
        tf.logging.info('score: %.2f', overall_score)

        _X, _y = X[y == 0], y[y == 0]
        tf.logging.info('score over non-van Gogh\'s: %.2f (%i samples)',
                        clf.score(_X, _y), _X.shape[0])

        _X, _y = X[y == 1], y[y == 1]
        tf.logging.info('score over van Gogh\'s: %.2f (%i samples)',
                        clf.score(_X, _y), _X.shape[0])

        return overall_score


if __name__ == '__main__':
    args = arg_parser.parse_args()

    print(__doc__, flush=True)

    (ExperimentSet(experiment_cls=StraightPredictionExperiment)
     .load_from_json(args.constants)
     .run())
