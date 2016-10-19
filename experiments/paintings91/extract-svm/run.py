"""Transform and Predict on Paintings91 dataset.

Uses an network trained over `imagenet` to transform paintings in
Paintings91 dataset into their low-dimensional representations.
Finally, exploits LinearSVM to classify these paintings.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import os

import numpy as np
import tensorflow as tf
from artificial.utils.experiments import ExperimentSet, Experiment, arg_parser
from keras import applications, layers, backend as K
from keras.engine import Input, Model
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from connoisseur import datasets
from connoisseur.fusion import SkLearnFusion


class TransformAndPredictExperiment(Experiment):
    def setup(self):
        c = self.consts

        if c.model_file:
            os.makedirs(os.path.dirname(c.model_file), exist_ok=True)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        s = tf.Session(config=config)
        K.set_session(s)

        np.random.seed(c.seed)

    def run(self):
        c = self.consts

        tf.logging.info('loading Paintings91 data set...')
        paintings91 = datasets.Paintings91(c).download().extract().check().load()

        if c.base_model in ('InceptionV3', 'Xception'):
            # These ones have a different pre-process function.
            from keras.applications.inception_v3 import preprocess_input
        else:
            from keras.applications.imagenet_utils import preprocess_input

        with tf.device(c.device):
            tf.logging.info('host device: %s', c.device)

            images = Input(batch_shape=[None] + c.image_shape)
            base_model_cls = getattr(applications, c.base_model)
            base_model = base_model_cls(weights='imagenet',
                                        input_tensor=images,
                                        include_top=False)
            x = base_model.output

            if c.dense_arch == 'original' and c.base_model == 'InceptionV3':
                x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
                x = layers.Flatten(name='flatten')(x)
            elif c.dense_arch == 'original' and c.base_model == 'Xception':
                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            else:
                x = layers.Flatten(name='flatten')(x)

            model = Model(input=base_model.input, output=x)

        tf.logging.info('embedding...')
        X, y = paintings91.train_data

        # Patches don't matter right now.
        X = preprocess_input(X.reshape((-1,) + X.shape[2:]))
        X = model.predict(X, batch_size=c.batch_size)
        # Assign the label associated with every sample
        # to every patch of that sample.
        y = np.repeat(y, c.train_n_patches)

        X_test, y_test = paintings91.test_data
        X_test = model.predict(
            preprocess_input(
                X_test.reshape((-1,) + X_test.shape[2:])),
            batch_size=c.batch_size).reshape(X_test.shape[:2] + (-1,))

        del model, base_model, paintings91
        # Let's go ahead and delete the session already,
        # as we won't be using the cnn anymore.
        K.clear_session()

        tf.logging.info('training...')

        if c.grid_searching:
            flow = Pipeline([('pca', PCA()), ('svm', LinearSVC())])
            clf = GridSearchCV(estimator=flow, param_grid=c.grid_search_params, n_jobs=c.n_jobs, cv=5, verbose=1)
            clf.fit(X, y)
            tf.logging.info('best params: %s', clf.best_params_)
            tf.logging.info('best score on validation data: %.2f', clf.best_score_)
            pca = clf.best_estimator_.named_steps['pca']
        else:
            # These are the best parameters I've found so far.
            clf = Pipeline([('pca', PCA(n_components=.999, copy=False)),
                            ('svm', LinearSVC(C=.1, dual=X.shape[0] <= X.shape[1], class_weight='balanced'))])
            clf.fit(X, y)
            pca = clf.named_steps['pca']

        tf.logging.info('embedded onto the R^%i (variance maintained: %.2f)',
                        pca.n_components_, np.sum(pca.explained_variance_ratio_))
        tf.logging.info('score on training data: %.2f', clf.score(X, y))

        tf.logging.info('testing...')

        for strategy in ('farthest', 'sum', 'most_frequent'):
            f = SkLearnFusion(clf, strategy=strategy)
            tf.logging.info('score on testing data, using '
                            'strategy `%s`: %.2f',
                            strategy, f.score(X_test, y_test))

        if c.model_file:
            tf.logging.info('saving model snapshot...')
            joblib.dump(clf, c.model_file)


if __name__ == '__main__':
    args = arg_parser.parse_args()

    print(__doc__, flush=True)

    logging.basicConfig(level=logging.INFO, filename='./run.log')
    for logger in ('artificial', 'tensorflow'):
        logger = logging.getLogger(logger)
        logger.setLevel(logging.DEBUG)

    (ExperimentSet(experiment_cls=TransformAndPredictExperiment)
     .load_from_json(args.constants)
     .run())
