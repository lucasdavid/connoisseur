"""Transfer and fine-tune InceptionV3 on van Gogh dataset.

Uses InceptionV3 trained over `imagenet` and fine-tune it to van Gogh dataset.
Image patches are loaded directly from the disk. Finally, train an SVM over
the fine-tuned extraction network.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os

import tensorflow as tf
from keras import callbacks, optimizers, backend as K
from keras.applications import InceptionV3
from keras.engine import Input, Model
from keras.layers import Dense, Flatten, AveragePooling2D
from sacred import Experiment

from connoisseur import datasets, utils

ex = Experiment('4-1-train-inception')


@ex.config
def config():
    dataset_seed = 4
    batch_size = 64
    image_shape = [299, 299, 3]
    train_shuffle = True
    train_augmentations = []
    dataset_train_seed = 12
    valid_shuffle = True
    valid_augmentations = []
    dataset_valid_seed = 98
    valid_split = .3
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"

    inception_optimal_params = {'lr': 0.0001, }
    ckpt_file = './ckpt/inception{epoch:02d}-{val_loss:.2f}.hdf5'
    nb_epoch = 100
    train_samples_per_epoch = 26048
    nb_val_samples = 8136
    nb_worker = 8
    early_stop_patience = 30
    tensorboard_file = './logs/long-training'


@ex.automain
def run(dataset_seed, image_shape, batch_size, data_dir,
        train_shuffle, train_augmentations, dataset_train_seed,
        valid_shuffle, valid_augmentations, dataset_valid_seed,
        valid_split,

        device, inception_optimal_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, tensorboard_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

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
        shuffle=valid_shuffle, seed=dataset_valid_seed)

    print('building model...')
    with tf.device(device):
        images = Input(batch_shape=[None] + image_shape)
        base_model = InceptionV3(weights='imagenet', input_tensor=images, include_top=False)
        x = base_model.output
        x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(2, activation='softmax', name='predictions')(x)

        model = Model(input=base_model.input, output=x)

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
                callbacks.ModelCheckpoint(ckpt_file, period=20, verbose=1),
            ])
        print('training completed.')
    except KeyboardInterrupt:
        print('training interrupted by user.')
