"""3 Train Contrastive.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from sacred import Experiment

from base import combine_pairs_for_training_gen, evaluate, build_model

ex = Experiment('3-train-contrastive')


@ex.config
def config():
    batch_size = 250
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"

    opt_params = {'lr': 0.001}
    ckpt_file = './ckpt/opt-weights.hdf5'
    nb_epoch = 1000
    train_samples_per_epoch = 24000
    nb_val_samples = 2048
    nb_worker = 1
    early_stop_patience = 200
    reduce_lr_on_plateau_patience = 100
    tensorboard_file = './logs/loss:contrastive,fc:3,samples:2048-dropout:.5'


@ex.automain
def run(batch_size, data_dir,
        device, opt_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, reduce_lr_on_plateau_patience, tensorboard_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    data = {}
    for phase in ('train', 'valid'):
        try:
            with open(os.path.join(data_dir, 'vgdb_2016', '%s.pickle' % phase), 'rb') as f:
                data[phase] = pickle.load(f)
                # Data as-it-is and categorical labels.
                data[phase] = (data[phase]['data'],
                               np.argmax(data[phase]['target'], axis=-1),
                               np.array(data[phase]['names'], copy=False))
        except IOError:
            continue

    if 'valid' not in data:
        # Separate train and valid sets.
        X, y, names = data['train']
        labels = np.unique(y)
        indices = [np.where(y == label)[0] for label in labels]
        for i in indices: np.random.shuffle(i)
        n_val_samples = int(nb_val_samples // len(labels))
        train_indices = np.concatenate([p[n_val_samples:] for p in indices])
        valid_indices = np.concatenate([p[:n_val_samples] for p in indices])

        data['train'] = X[train_indices], y[train_indices], names[train_indices]
        data['valid'] = X[valid_indices], y[valid_indices], names[valid_indices]

    print('train-valid split:', data['train'][0].shape[0], data['valid'][0].shape[0])

    train_data = combine_pairs_for_training_gen(*data['train'], batch_size=batch_size)
    valid_data = combine_pairs_for_training_gen(*data['valid'], batch_size=batch_size)

    model = build_model(x_shape=(2048,), opt_params=opt_params, device=device, compile_opt=True,
                        dropout_prob=.5)

    print('training network...')
    try:
        model.fit_generator(
            train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
            validation_data=valid_data, nb_val_samples=nb_val_samples, nb_worker=nb_worker,
            callbacks=[
                callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=reduce_lr_on_plateau_patience),
                callbacks.EarlyStopping(patience=early_stop_patience),
                callbacks.TensorBoard(tensorboard_file, histogram_freq=5),
                callbacks.ModelCheckpoint(ckpt_file, save_best_only=True, verbose=1),
            ])
    except KeyboardInterrupt:
        print('training interrupted by user.')
