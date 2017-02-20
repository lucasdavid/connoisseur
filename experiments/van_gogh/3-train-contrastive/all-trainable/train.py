"""3 Train Contrastive.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from sacred import Experiment

from connoisseur.utils.image import PairsDirectoryIterator, ImageDataGenerator

from base import combine_pairs_for_training_gen, evaluate
from model import build_model

ex = Experiment('3-train-contrastive-training-convolutions')


@ex.config
def config():
    image_shape = [299, 299, 3]
    batch_size = 64
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"

    train_augmentations = []
    train_shuffle = True
    train_dataset_seed = 14
    valid_augmentations = []
    valid_shuffle = True
    valid_dataset_seed = 9

    opt_params = {'lr': 0.001}
    ckpt_file = './ckpt/opt-weights.hdf5'
    nb_epoch = 500
    train_samples_per_epoch = 24000
    nb_val_samples = 2048
    nb_worker = 1
    early_stop_patience = 80
    reduce_lr_on_plateau_patience = 10
    tensorboard_file = './logs/loss:contrastive,fc:3,samples:2048-dropout:.5'


@ex.automain
def run(batch_size, data_dir,
        image_shape,
        train_augmentations, train_shuffle, train_dataset_seed,
        valid_augmentations, valid_shuffle, valid_dataset_seed,
        device, opt_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, reduce_lr_on_plateau_patience, tensorboard_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    g = ImageDataGenerator(rescale=2. / 255., featurewise_center=True)
    g.mean = 1
    train_data = PairsDirectoryIterator(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'train'),
        image_data_generator=g, target_size=image_shape[:2],
        augmentations=train_augmentations, batch_size=batch_size,
        shuffle=train_shuffle, seed=train_dataset_seed)

    valid_data = PairsDirectoryIterator(
        os.path.join(data_dir, 'vgdb_2016', 'extracted_patches', 'valid'),
        image_data_generator=g, target_size=image_shape[:2],
        augmentations=valid_augmentations, batch_size=batch_size,
        shuffle=valid_shuffle, seed=valid_dataset_seed)

    data = {}
    for phase in ('train', 'valid', 'test'):
        try:
            with open(os.path.join(data_dir, 'vgdb_2016', '%s.pickle' % phase), 'rb') as f:
                data[phase] = pickle.load(f)
                # Data as-it-is and categorical labels.
                data[phase] = (data[phase]['data'],
                               np.argmax(data[phase]['target'], axis=-1),
                               np.array(data[phase]['names'], copy=False))
        except IOError:
            continue

    # Build using convolutions, because we are reading inputting images!
    model = build_model(x_shape=image_shape, opt_params=opt_params,
                        device=device, compile_opt=True, dropout_prob=.5,
                        convolutions=True)

    print('training network...')
    try:
        model.fit_generator(
            train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
            validation_data=valid_data, nb_val_samples=nb_val_samples, nb_worker=nb_worker,
            callbacks=[
                callbacks.ReduceLROnPlateau(patience=reduce_lr_on_plateau_patience),
                # callbacks.EarlyStopping(patience=early_stop_patience),
                callbacks.TensorBoard(tensorboard_file, write_graph=False),
                callbacks.ModelCheckpoint(ckpt_file, verbose=1, save_best_only=True),
            ])
    except KeyboardInterrupt:
        print('training interrupted by user.')
