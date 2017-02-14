"""Inception PCA SVM Disk.

This experiment consists on the following procedures:

 * Extract features from paintings using InceptionV3 pre-trained over imagenet
 * Train an SVM over the extracted features
 * Classify each patch of each test painting with the trained SVM
 * Fuse the predicted classes for each patch, resulting in the predicted class for the painting

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import os

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
from connoisseur.utils.image import ImageDataGenerator

ex = Experiment('inception-pca-svm-disk')


@ex.config
def config():
    svm_seed = 2
    dataset_seed = 4
    classes = None
    n_train_samples = 7936
    batch_size = 64
    image_shape = [299, 299, 3]
    train_augmentations = []
    train_shuffle = True
    train_dataset_seed = 14
    test_n_patches = 40
    test_augmentations = []
    test_shuffle = True
    device = "/gpu:1"
    n_jobs = 8
    data_dir = "/datasets/ldavid/van_gogh"
    hard_balancing = False
    grid_searching = False
    param_grid = {'svc__C': [0.1, 1, 10, 100],
                  'svc__kernel': ['rbf', 'linear'],
                  'svc__class_weight': ['balanced', None]}


@ex.automain
def run(_run, dataset_seed, svm_seed, n_train_samples,
        image_shape, batch_size, data_dir,
        train_augmentations, train_dataset_seed, train_shuffle,
        test_n_patches, test_augmentations,
        hard_balancing,
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
        test_n_patches=test_n_patches,
        test_augmentations=test_augmentations,
        random_state=dataset_seed
    ).download().extract().split_train_valid().extract_patches_to_disk()

    g = ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1
    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'train'),
        target_size=image_shape[:2],
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=train_shuffle, seed=train_dataset_seed)

    X, y = [], []
    n_samples = 0
    while n_samples < n_train_samples:
        _X, _y = next(train_data)
        _X = f_model.predict(_X, batch_size=batch_size)
        X.append(_X), y.append(_y)
        n_samples += batch_size

    X, y = map(np.concatenate, (X, y))
    y = np.argmax(y, axis=1)

    # Hard balance classes.
    _, occurrences = np.unique(y, return_counts=True)
    print('label occurrences before balancing: %s' % dict(enumerate(occurrences)))

    if hard_balancing:
        min_label_occurrence = occurrences.min()

        X_balanced, y_balanced = [], []
        for label_code in range(occurrences.shape[0]):
            matcher = y == label_code
            X_balanced.append(X[matcher][:min_label_occurrence])
            y_balanced.append(y[matcher][:min_label_occurrence])

        X, y = map(np.concatenate, (X_balanced, y_balanced))
        del X_balanced, y_balanced

        _, occurrences = np.unique(y, return_counts=True)
        print('label occurrences after balancing: %s' % dict(enumerate(occurrences)))
        print('train data: %s, %f MB' % (X.shape, X.nbytes / 1024 / 1024))

    X_test, y_test = vangogh.load_patches_from_full_images('test').test_data
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
