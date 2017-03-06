"""5 Train Contrastive.

Train or fine-tune InceptionV3 architecture to fit van-Gogh data set.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os

import tensorflow as tf
from keras import backend as K, callbacks, optimizers
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from connoisseur.models import build_siamese_model
from connoisseur.utils.image import PairsDirectoryIterator, ImageDataGenerator

from base import contrastive_loss

ex = Experiment('5-train-contrastive')

ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    image_shape = [299, 299, 3]
    batch_size = 150
    device = '/gpu:0'
    data_dir = '/datasets/van_gogh'
    train_dir = '/datasets/van_gogh/vgdb_2016/extracted_patches_256/train'
    valid_dir = '/datasets/van_gogh/vgdb_2016/extracted_patches_256/valid'
    balanced_classes = True

    train_augmentations = []
    train_shuffle = True
    train_dataset_seed = 14
    valid_augmentations = []
    valid_shuffle = True
    valid_dataset_seed = 9

    arch = 'inejc'
    network_weights = None
    pre_trained_weights = None

    opt_params = {'lr': 0.005}
    ckpt_file = '/work/van_gogh/5-train-contrastive/ckpt/optimal.hdf5'
    nb_epoch = 10000
    train_samples_per_epoch = 13350
    nb_val_samples = 2048
    nb_worker = 1
    early_stop_patience = 300
    reduce_lr_on_plateau_patience = 100
    tensorboard_file = './logs/loss:contrastive'

    initial_epoch = 0


@ex.automain
def run(batch_size, data_dir, train_dir, valid_dir, balanced_classes,
        image_shape,
        train_augmentations, train_shuffle, train_dataset_seed,
        valid_augmentations, valid_shuffle, valid_dataset_seed,
        device, opt_params, ckpt_file, pre_trained_weights,
        arch, network_weights,
        train_samples_per_epoch, nb_epoch, nb_val_samples, nb_worker,
        early_stop_patience, reduce_lr_on_plateau_patience, tensorboard_file,
        initial_epoch):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    g = ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        height_shift_range=.2,
        width_shift_range=.2,
        rescale=2. / 255.,
        fill_mode='reflect')
    g.mean = 1.
    train_data = PairsDirectoryIterator(
        train_dir,
        image_data_generator=g, target_size=image_shape[:2],
        augmentations=train_augmentations, batch_size=batch_size,
        balanced_classes=balanced_classes, shuffle=train_shuffle,
        seed=train_dataset_seed)
    valid_data = PairsDirectoryIterator(
        valid_dir,
        image_data_generator=g, target_size=image_shape[:2],
        augmentations=valid_augmentations, batch_size=batch_size,
        balanced_classes=balanced_classes, shuffle=valid_shuffle,
        seed=valid_dataset_seed)

    with tf.device(device):
        model = build_siamese_model(x_shape=image_shape, arch=arch, weights=network_weights)
        opt = optimizers.Adam(**opt_params)
        model.compile(optimizer=opt, loss=contrastive_loss)

        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
        elif initial_epoch > 0:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        print('training network...')
        try:
            model.fit_generator(
                train_data, samples_per_epoch=train_samples_per_epoch,
                nb_epoch=nb_epoch,
                validation_data=valid_data, nb_val_samples=nb_val_samples,
                nb_worker=nb_worker,
                callbacks=[
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=reduce_lr_on_plateau_patience),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file, histogram_freq=5),
                    callbacks.ModelCheckpoint(ckpt_file, verbose=1, save_best_only=True),
                ],
                initial_epoch=initial_epoch)
        except KeyboardInterrupt:
            print('training interrupted by user.')
        else:
            print('done.')
