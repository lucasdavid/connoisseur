"""Use a network to embed samples from the van Gogh dataset, then classify
them using an SVM.

Uses a trained network to embed samples from van Gogh.
Image patches are loaded directly from the disk. Finally, train an SVM over
the fine-tuned extraction network.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import preprocess_input
from keras.engine import Model
from sacred import Experiment
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from connoisseur import datasets
from connoisseur.fusion import SkLearnFusion
from connoisseur.models.painter_by_numbers import build_model

ex = Experiment('painter-by-numbers.4-train.train-svm')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 64
    image_shape = [299, 299, 3]
    train_shuffle = True
    train_n_patches = 40
    train_augmentations = []
    dataset_train_seed = 12
    valid_shuffle = True
    valid_n_patches = 40
    valid_augmentations = []
    dataset_valid_seed = 98
    valid_split = .3
    test_shuffle = True
    test_n_patches = 80
    dataset_test_seed = 53
    test_augmentations = []
    device = "/gpu:0"
    data_dir = "/datasets/vangogh"

    architecture = 'inception'
    opt_ckpt_file = './ckpt/opt-weights.hdf5'

    svm_seed = 2
    grid_searching = False
    param_grid = {'svc__C': [0.1, 1, 10, 100],
                  'svc__kernel': ['rbf', 'linear'],
                  'svc__class_weight': ['balanced', None]}
    svm_ckpt_file = './ckpt/optimal-svm.pkl'
    n_jobs = 8


def load_and_embed(model, dataset, phase, batch_size):
    (X, y, names), = dataset.load_patches_from_full_images(phase)
    preprocess_input(X)
    X = (model.predict(X.reshape((-1,) + X.shape[2:]), batch_size=batch_size)
         .reshape(X.shape[:2] + (-1,)))
    return X, y, names


@ex.automain
def run(_run, dataset_seed,
        image_shape, batch_size, data_dir,
        train_shuffle, train_n_patches, train_augmentations,
        test_shuffle, test_n_patches, test_augmentations,
        valid_n_patches, valid_augmentations, valid_split,

        device, opt_ckpt_file,
        architecture,
        
        svm_ckpt_file, svm_seed, grid_searching, param_grid, n_jobs):
    os.makedirs(os.path.dirname(svm_ckpt_file), exist_ok=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    vangogh = datasets.VanGogh(
        base_dir=data_dir, image_shape=image_shape,
        train_n_patches=train_n_patches,
        test_n_patches=test_n_patches,
        valid_n_patches=valid_n_patches,
        train_augmentations=train_augmentations,
        test_augmentations=test_augmentations,
        valid_augmentations=valid_augmentations,
        random_state=dataset_seed)

    with tf.device(device):
        print('building model...')
        model = build_model(image_shape, arch=architecture, weights=None, dropout_p=0)
        extract_model = Model(input=model.input, output=model.get_layer('flatten').output)
        # Restore best parameters.
        print('loading weights from:', opt_ckpt_file)
        model.load_weights(opt_ckpt_file)

    X, y, names = load_and_embed(extract_model, vangogh, 'train', batch_size)
    X_valid, y_valid, names_valid = load_and_embed(extract_model, vangogh, 'valid', batch_size)
    X, y = (X.reshape((-1,) + X.shape[2:]), np.repeat(y, train_n_patches))
    X_valid, y_valid = (X_valid.reshape((-1,) + X_valid.shape[2:]), np.repeat(y_valid, valid_n_patches))
    X, y = np.concatenate((X, X_valid)), np.concatenate((y, y_valid))
    del X_valid, y_valid

    X_test, y_test, names_test = load_and_embed(extract_model, vangogh, 'test', batch_size)
    del model, extract_model, vangogh
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

    joblib.dump(model, svm_ckpt_file)

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
