"""Transform With Inception.

This experiment consists on the following procedures:

 * Load each painting patch and transform it using InceptionV3.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from keras import layers, backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.engine import Input, Model
from sacred import Experiment

from connoisseur import datasets
from connoisseur.utils.image import ImageDataGenerator

ex = Experiment('2-transform-inception')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 256
    image_shape = [299, 299, 3]
    train_shuffle = True
    train_n_patches = 2
    train_augmentations = []
    dataset_train_seed = 12
    valid_shuffle = True
    valid_n_patches = 2
    valid_augmentations = []
    dataset_valid_seed = 98
    test_shuffle = True
    test_n_patches = 80
    dataset_test_seed = 53
    test_augmentations = []
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/paintings91"


@ex.automain
def run(image_shape, batch_size, data_dir, dataset_seed,
        train_shuffle, train_n_patches, train_augmentations, dataset_train_seed,
        valid_shuffle, valid_n_patches, valid_augmentations, dataset_valid_seed,
        test_shuffle, test_n_patches, test_augmentations, dataset_test_seed,
        device):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    with tf.device(device):
        base_model = InceptionV3(weights='imagenet', input_shape=image_shape, include_top=False)
        x = base_model.output
        x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = layers.Flatten(name='flatten')(x)
        f_model = Model(input=base_model.input, output=x)

    vangogh = (datasets.Paintings91(
        base_dir=data_dir, image_shape=image_shape,
        train_n_patches=train_n_patches,
        test_n_patches=test_n_patches,
        train_augmentations=train_augmentations,
        test_augmentations=test_augmentations,
        random_state=dataset_seed)
               .download()
               .extract()
               .split_train_test()
               .extract_patches_to_disk())

    g = ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1
    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'Paintings91', 'extracted_patches', 'train'),
        target_size=image_shape[:2], class_mode='categorical',
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=train_shuffle, seed=dataset_train_seed)

    test_data = g.flow_from_directory(
        os.path.join(data_dir, 'Paintings91', 'extracted_patches', 'test'),
        target_size=image_shape[:2], class_mode='categorical',
        augmentations=test_augmentations, batch_size=batch_size,
        shuffle=test_shuffle, seed=dataset_test_seed)

    for phase in ('train', 'test'):
        data = locals()['%s_data' % phase]

        print('transforming %i %s samples' % (data.N, phase), end='')

        total_samples_seen = 0
        X, y = [], []

        while total_samples_seen < data.N:
            _X, _y = next(data)
            _X = f_model.predict_on_batch(_X)

            X.append(_X)
            y.append(_y)
            total_samples_seen += _X.shape[0]
            print('.', end='')

        X, y = np.concatenate(X), np.concatenate(y)
        with open(os.path.join(data_dir, 'Paintings91', '%s.pickle' % phase), 'wb') as f:
            pickle.dump({'data': X, 'target': y}, f, pickle.HIGHEST_PROTOCOL)

        print('done.')
