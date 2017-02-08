"""Inception Siamese.

This experiment consists on the following procedures:

 * Extract features from pairs of paintings using InceptionV3 pre-trained over imagenet
 * Train an SVM over the extracted features
 * Classify each patch of each test painting with the trained SVM
 * Fuse the predicted classes for each patch, resulting in the predicted class for the painting

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import itertools
import os

import numpy as np
import tensorflow as tf
from keras import callbacks, layers, backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.engine import Input, Model
from sacred import Experiment

from connoisseur import datasets
from connoisseur.utils.image import ImageDataGenerator

ex = Experiment('train-inception-siamese-contrastive-loss')


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
    ckpt_file = './ckpt/weights.{epoch:d}-{val_loss:.2f}.hdf5'
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


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def combined_pairs_flow(flow):
    while True:
        X, y = next(flow)
        y_enc = np.argmax(y, axis=-1)

        combinations = np.array(list(itertools.combinations(range(X.shape[0]), 2)))
        y_combined = np.logical_and(*y_enc[combinations].T)
        yield X[combinations].transpose(1, 0, 2, 3, 4), y_combined


def build_model(image_shape, device):
    with tf.device(device):
        base_model = InceptionV3(weights=None, input_shape=image_shape,
                                 include_top=False)
        x = base_model.output
        x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = layers.Flatten(name='flatten')(x)
        base_model = Model(input=base_model.input, output=x)

        img_a = Input(batch_shape=[None] + image_shape)
        img_b = Input(batch_shape=[None] + image_shape)
        leg_a = base_model(img_a)
        leg_b = base_model(img_b)

        x = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([leg_a, leg_b])
        model = Model(input=[img_a, img_b], output=x)
        model.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])

    return model


@ex.automain
def run(_run, dataset_seed,
        image_shape, batch_size, data_dir,
        train_shuffle, train_n_patches, train_augmentations, dataset_train_seed,
        test_shuffle, test_n_patches, test_augmentations, dataset_valid_seed,
        valid_shuffle, valid_n_patches, valid_augmentations, valid_split, dataset_test_seed,

        device, inception_optimal_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, tensorboard_file,

        nb_test_samples,

        svm_seed, grid_searching, param_grid, svc_optimal_params, n_jobs):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    s = tf.Session(config=config)
    K.set_session(s)

    model = build_model(image_shape=image_shape, device=device)

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
    ).download().extract().check().extract_patches_to_disk()

    g = ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1
    train_data = combined_pairs_flow(g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'train'),
        target_size=image_shape[:2],
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=train_shuffle, seed=dataset_train_seed))

    valid_data = combined_pairs_flow(g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'valid'),
        target_size=image_shape[:2],
        augmentations=valid_augmentations, batch_size=batch_size,
        shuffle=valid_shuffle, seed=dataset_valid_seed))

    test_data = combined_pairs_flow(g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'test'),
        target_size=image_shape[:2],
        augmentations=test_augmentations, batch_size=batch_size,
        shuffle=test_shuffle, seed=dataset_test_seed))

    print('training siamese inception...')
    try:
        model.fit_generator(
            generator=train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
            validation_data=valid_data, nb_val_samples=nb_val_samples,
            nb_worker=nb_worker, verbose=1,
            callbacks=[
                callbacks.EarlyStopping(patience=early_stop_patience),
                callbacks.TensorBoard(tensorboard_file, write_graph=False),
                callbacks.ModelCheckpoint(ckpt_file, verbose=1),
            ])
        print('training completed.')
    except KeyboardInterrupt:
        print('training interrupted by user.')
