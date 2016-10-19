"""Transfer and fine-tune InceptionV3 on van Gogh dataset, then classify
using an SVM.

Uses InceptionV3 trained over `imagenet` and fine-tune it to van Gogh dataset.
Image patches are loaded directly from the disk. Finally, train an SVM over
the fine-tuned extraction network.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os

import numpy as np
import tensorflow as tf
from keras import callbacks, optimizers, backend as K
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.engine import Input, Model
from keras.layers import Dense, Flatten, AveragePooling2D
from sacred import Experiment
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from connoisseur import datasets, utils
from connoisseur.fusion import SkLearnFusion

ex = Experiment('train-inception-pca-train-svm')


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
    valid_split = .2
    test_shuffle = True
    test_n_patches = 80
    dataset_test_seed = 53
    test_augmentations = []
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"

    inception_optimal_params = {'lr': 0.0001, }
    ckpt_file = './ckpt/inceptionV3.hdf5'
    train_samples_per_epoch = 1000
    nb_epoch = 100
    nb_val_samples = 670
    nb_worker = 8
    early_stop_patience = 10
    tensorboard_file = './logs/base-training'
    nb_test_samples = 670

    svm_seed = 2
    grid_searching = False
    param_grid = {'svc__C': [0.1, 1, 10, 100],
                  'svc__kernel': ['rbf', 'linear'],
                  'svc__class_weight': ['balanced', None]}
    svc_optimal_params = {'class_weight': 'balanced'}
    n_jobs = 8


def load_and_embed(model, dataset, phase, batch_size):
    X, y = dataset.load(phase).get(phase)
    dataset.unload(phase)
    preprocess_input(X)
    X = (model.predict(X.reshape((-1,) + X.shape[2:]), batch_size=batch_size)
         .reshape(X.shape[:2] + (-1,)))
    dataset.unload(phase)
    return X, y


@ex.automain
def run(_run, dataset_seed,
        image_shape, batch_size, data_dir,
        train_shuffle, train_n_patches, train_augmentations, dataset_train_seed,
        test_shuffle, test_n_patches, test_augmentations, dataset_valid_seed,
        val_shuffle, valid_n_patches, valid_augmentations, valid_split, dataset_test_seed,

        device, inception_optimal_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, tensorboard_file,

        nb_test_samples,

        svm_seed, grid_searching, param_grid, svc_optimal_params, n_jobs):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
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
        valid_split=valid_split,
        random_state=dataset_seed
    ).download().extract().check()

    # These parameters produce the same results of `preprocess_input`.
    g = utils.image.ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1
    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'train'),
        target_size=image_shape[:2],
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=train_shuffle, seed=dataset_train_seed)

    val_data = g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'valid'),
        target_size=image_shape[:2],
        augmentations=valid_augmentations, batch_size=batch_size,
        shuffle=val_shuffle, seed=dataset_valid_seed)

    test_data = g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'test'),
        target_size=image_shape[:2],
        augmentations=test_augmentations, batch_size=batch_size,
        shuffle=test_shuffle, seed=dataset_test_seed)

    print('building model...')
    with tf.device(device):
        images = Input(batch_shape=[None] + image_shape)
        base_model = InceptionV3(weights='imagenet', input_tensor=images, include_top=False)
        x = base_model.output
        x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

        model = Model(input=base_model.input, output=x)
        extract_model = Model(input=model.input, output=model.get_layer('flatten').output)

        opt = optimizers.Adam(**inception_optimal_params)
        model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

    print('training InceptionV3...')
    try:
        model.fit_generator(
            generator=train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
            validation_data=val_data, nb_val_samples=nb_val_samples,
            nb_worker=nb_worker, verbose=1,
            callbacks=[
                callbacks.EarlyStopping(patience=early_stop_patience),
                callbacks.TensorBoard(tensorboard_file, write_graph=False),
                callbacks.ModelCheckpoint(ckpt_file, save_best_only=True, verbose=1),
            ])
        print('training completed.')
    except KeyboardInterrupt:
        print('training interrupted by user.')

    # Restore best parameters.
    print('loading weights from: %s', ckpt_file)
    model.load_weights(ckpt_file)

    scores = model.evaluate_generator(test_data, nb_test_samples, nb_worker=nb_worker)
    print('soft-max classifier score on test data: %s', dict(zip(model.metrics_names, scores)))

    X, y = load_and_embed(extract_model, vangogh, 'train', train_n_patches)
    X_valid, y_valid = load_and_embed(extract_model, vangogh, 'valid', valid_n_patches)
    X, y = (X.reshape((-1,) + X.shape[2:]), np.repeat(y, train_n_patches))
    X_valid, y_valid = (X_valid.reshape((-1,) + X_valid.shape[2:]), np.repeat(y_valid, valid_n_patches))
    X, y = np.concatenate((X, X_valid)), np.concatenate((y, y_valid))
    del X_valid, y_valid

    X_test, y_test = load_and_embed(extract_model, vangogh, 'test', test_n_patches)
    del base_model, model, extract_model, vangogh
    K.clear_session()

    if grid_searching:
        grid = GridSearchCV(
            estimator=SVC(class_weight='balanced', random_state=svm_seed),
            param_grid=param_grid,
            n_jobs=n_jobs
        )
        grid.fit(X, y)

        print('best parameters: %s', grid.best_params_)
        print('validation score for parameters: %f', grid.best_score_)

        clf = grid.best_estimator_
    else:
        clf = SVC(**svc_optimal_params, random_state=svm_seed)
        clf.fit(X, y)
    del X, y

    for strategy in ('farthest', 'sum', 'most_frequent'):
        f = SkLearnFusion(clf, strategy=strategy)
        p_test = f.predict(X_test)
        accuracy_score = metrics.accuracy_score(y_test, p_test)
        print('score using', strategy, 'strategy: %.2f' % accuracy_score, '\n',
              metrics.classification_report(y_test, p_test), '\nConfusion matrix:\n',
              metrics.confusion_matrix(y_test, p_test))

        if accuracy_score > _run.info['accuracy']['test']:
            _run.info['accuracy']['test'] = accuracy_score
