"""Inception Siamese.

This experiment consists on the following procedures:

 * Extract features from pairs of paintings using InceptionV3 pre-trained over imagenet
 * Train an SVM over the extracted features
 * Classify each patch of each test painting with the trained SVM
 * Fuse the predicted classes for each patch, resulting in the predicted class for the painting

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os

import tensorflow as tf
from keras import callbacks, optimizers, backend as K
from keras.engine import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda
from keras.models import Sequential
from sacred import Experiment

from connoisseur import datasets
from connoisseur.utils.image import ImageDataGenerator

ex = Experiment('siamese-contrastive-loss')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 30
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

    opt_params = {'lr': 0.0001, 'momentum': .9, 'decay': 1e-6, 'nesterov': True}
    ckpt_file = './ckpt/weights.{epoch:d}-{val_loss:.2f}.hdf5'
    train_samples_per_epoch = 1000
    nb_epoch = 100
    nb_val_samples = 670
    nb_worker = 4
    early_stop_patience = 10
    tensorboard_file = './logs/siamese-contrastive-loss-1'


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


def compute_accuracy(predictions, labels):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return labels[predictions.ravel() < 0.5].mean()


def build_model(x_shape, opt_params, device):
    with tf.device(device):
        base_network = Sequential([
            Conv2D(64, 3, 3, activation='relu', name='block1_conv1', input_shape=x_shape),
            Conv2D(64, 3, 3, activation='relu', name='block1_conv2'),
            MaxPooling2D((4, 4), strides=(2, 2), name='block1_pool'),

            Conv2D(128, 3, 3, activation='relu', name='block2_conv1'),
            Conv2D(128, 3, 3, activation='relu', name='block2_conv2'),
            MaxPooling2D((4, 4), strides=(2, 2), name='block2_pool'),

            Conv2D(256, 3, 3, activation='relu', name='block3_conv1'),
            Conv2D(256, 3, 3, activation='relu', name='block3_conv2'),
            Conv2D(256, 3, 3, activation='relu', name='block3_conv3'),
            MaxPooling2D((4, 4), strides=(2, 2), name='block3_pool'),

            Flatten(name='flatten'),

            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(1024, activation='relu'),
        ])

        img_a = Input(shape=x_shape)
        img_b = Input(shape=x_shape)
        leg_a = base_network(img_a)
        leg_b = base_network(img_b)

        x = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([leg_a, leg_b])
        model = Model(input=[img_a, img_b], output=x)
        opt = optimizers.SGD(**opt_params)
        model.compile(optimizer=opt, loss=contrastive_loss)

    return model


@ex.automain
def run(_run, dataset_seed,
        image_shape, batch_size, data_dir,
        train_shuffle, train_n_patches, train_augmentations, dataset_train_seed,
        test_shuffle, test_n_patches, test_augmentations, dataset_valid_seed,
        valid_shuffle, valid_n_patches, valid_augmentations, valid_split, dataset_test_seed,

        device, opt_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, tensorboard_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    K.set_session(sess)

    model = build_model(x_shape=image_shape, opt_params=opt_params, device=device)

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

    g = ImageDataGenerator(rescale=1. / 255.)
    train_data = g.flow_pairs_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'train'),
        target_size=image_shape[:2], class_mode='sparse',
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=train_shuffle, seed=dataset_train_seed)

    valid_data = g.flow_pairs_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'valid'),
        target_size=image_shape[:2], class_mode='sparse',
        augmentations=valid_augmentations, batch_size=batch_size,
        shuffle=valid_shuffle, seed=dataset_valid_seed)

    test_data = g.flow_pairs_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'test'),
        target_size=image_shape[:2], class_mode='sparse',
        augmentations=test_augmentations, batch_size=batch_size,
        shuffle=test_shuffle, seed=dataset_test_seed)

    print('training siamese inception...')

    try:
        with tf.device('/cpu:0'):
            model.fit_generator(
                generator=train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
                validation_data=valid_data, nb_val_samples=nb_val_samples,
                nb_worker=nb_worker, verbose=1,
                callbacks=[
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file, write_graph=False),
                    callbacks.ModelCheckpoint(ckpt_file, verbose=1, save_best_only=True),
                ])
            print('training completed.')
    except KeyboardInterrupt:
        print('training interrupted by user.')

    print('testing...')
    test_accuracy = 0
    n_batches = test_data.N // test_data.batch_size
    for batch in range(n_batches):
        X, y = next(test_data)
        p = model.predict_on_batch(X)
        test_accuracy += compute_accuracy(p, y)
    print('accuracy:', test_accuracy / n_batches)
