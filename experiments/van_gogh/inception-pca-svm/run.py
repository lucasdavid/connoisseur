"""Inception PCA SVM.

This experiment consists on the following procedures:

 * Extract features from paintings using InceptionV3 pre-trained over imagenet
 * Train an SVM over the extracted features
 * Classify each patch of each test painting with the trained SVM
 * Fuse the predicted classes for each patch, resulting in the predicted class for the painting

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import numpy as np
import tensorflow as tf
from keras import layers, backend as K
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.engine import Input, Model
from sacred import Experiment
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from connoisseur import datasets
from connoisseur.fusion import SkLearnFusion

ex = Experiment('inception-pca-svm')


@ex.config
def config():
    svm_seed = 2
    dataset_seed = 4
    batch_size = 64
    image_shape = [299, 299, 3]
    train_n_patches = 20
    train_augmentations = []
    test_n_patches = 20
    test_augmentations = []
    device = "/gpu:0"
    n_jobs = 8
    data_dir = "/datasets/ldavid/van_gogh"
    grid_searching = False
    param_grid = {'svc__C': [0.1, 1, 10, 100],
                  'svc__kernel': ['rbf', 'linear'],
                  'svc__class_weight': ['balanced', None]}


@ex.automain
def run(_run, dataset_seed, svm_seed,
        image_shape, batch_size, data_dir,
        train_n_patches, train_augmentations,
        test_n_patches, test_augmentations,
        device, grid_searching, param_grid, n_jobs):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    K.set_session(s)

    with tf.device(device):
        images = Input(batch_shape=[None] + image_shape)
        base_model = InceptionV3(weights='imagenet', input_tensor=images, include_top=False)
        x = base_model.output
        x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = layers.Flatten(name='flatten')(x)
        f_model = Model(input=base_model.input, output=x)

    vangogh = datasets.VanGogh(
        base_dir=data_dir, image_shape=image_shape,
        train_n_patches=train_n_patches,
        test_n_patches=test_n_patches,
        train_augmentations=train_augmentations,
        test_augmentations=test_augmentations,
        random_state=dataset_seed
    ).download().extract().check()

    X, y = vangogh.load('train').train_data
    vangogh.unload('train')
    preprocess_input(X)
    # Leave arrays flatten, we don't use patches for training.
    X = f_model.predict(X.reshape((-1,) + X.shape[2:]), batch_size=batch_size)
    y = np.repeat(y, train_n_patches)
    print('train data: %s, %f MB' % (X.shape, X.nbytes / 1024 / 1024))

    X_test, y_test = vangogh.load('test').test_data
    vangogh.unload('test')
    preprocess_input(X_test)
    X_test = (f_model.predict(X_test.reshape((-1,) + X_test.shape[2:]), batch_size=batch_size)
              .reshape(X_test.shape[:2] + (-1,)))
    print('test data: %s, %f MB' % (X_test.shape, X_test.nbytes / 1024 / 1024))

    del f_model, base_model, vangogh
    K.clear_session()

    if grid_searching:
        print('grid searching...')
        flow = Pipeline([
            ('pca', PCA(n_components=.99)),
            ('svc', SVC(class_weight='balanced', random_state=svm_seed))
        ])
        grid = GridSearchCV(estimator=flow, param_grid=param_grid, n_jobs=n_jobs, verbose=1)
        grid.fit(X, y)
        print('best params: %s' % grid.best_params_)
        print('best score on validation data: %.2f' % grid.best_score_)
        model = grid.best_estimator_
    else:
        print('training...')
        # These are the best parameters I've found so far.
        model = Pipeline([
            ('pca', PCA(n_components=.99)),
            ('svc', SVC(class_weight='balanced', random_state=svm_seed))
        ])
        model.fit(X, y)

    pca = model.named_steps['pca']
    print('dimensionality reduction: R^%i->R^%i (%.2f variance kept)'
          % (X.shape[-1], pca.n_components_, np.sum(pca.explained_variance_ratio_)))
    accuracy_score = model.score(X, y)
    print('score on training data: %.2f' % accuracy_score)
    _run.info['accuracy'] = {
        'train': accuracy_score,
        'test': -1
    }
    del X, y

    for strategy in ('farthest', 'sum', 'most_frequent'):
        f = SkLearnFusion(model, strategy=strategy)
        p_test = f.predict(X_test)
        accuracy_score = metrics.accuracy_score(y_test, p_test)
        print('score using', strategy, 'strategy: %.2f' % accuracy_score, '\n',
              metrics.classification_report(y_test, p_test), '\nConfusion matrix:\n',
              metrics.confusion_matrix(y_test, p_test))

        if accuracy_score > _run.info['accuracy']['test']:
            _run.info['accuracy']['test'] = accuracy_score
