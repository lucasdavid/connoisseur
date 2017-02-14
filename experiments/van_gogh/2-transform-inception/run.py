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
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"
    train_augmentations = []
    valid_augmentations = []
    test_augmentations = []


@ex.automain
def run(dataset_seed, image_shape, batch_size, device, data_dir,
        train_augmentations, valid_augmentations, test_augmentations):
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    with tf.device(device):
        images = Input(batch_shape=[None] + image_shape)
        base_model = InceptionV3(weights='imagenet', input_tensor=images,
                                 include_top=False)
        x = base_model.output
        x = layers.AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = layers.Flatten(name='flatten')(x)
        f_model = Model(input=base_model.input, output=x)

    (datasets.VanGogh(base_dir=data_dir, image_shape=image_shape,
                      n_jobs=8)
     .download()
     .extract()
     .extract_patches_to_disk())

    g = ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1
    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'train'),
        target_size=image_shape[:2], class_mode='categorical',
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=False, seed=dataset_seed)

    test_data = g.flow_from_directory(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'test'),
        target_size=image_shape[:2], class_mode='categorical',
        augmentations=test_augmentations, batch_size=batch_size,
        shuffle=False, seed=dataset_seed)

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
        with open(os.path.join(data_dir, 'vgdb_2016', '%s.pickle' % phase), 'wb') as f:
            pickle.dump({'data': X, 'target': y, 'names': data.filenames},
                        f, pickle.HIGHEST_PROTOCOL)

        print('done.')
