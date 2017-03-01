"""4 Train Triplet.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from keras import optimizers
from sacred import Experiment

from base import triplets_gen
from model import build_model, triplet_loss

ex = Experiment('4-train-triplet')


@ex.config
def config():
    batch_size = 250
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"

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
        device, opt_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, reduce_lr_on_plateau_patience, tensorboard_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    with tf.device(device):
        embedding_net, training_net = build_model(x_shape=(2048,),
                                                  dropout_prob=.5)

        opt = optimizers.Adam(**opt_params)
        training_net.compile(optimizer=opt, loss=triplet_loss)

    data = {}
    for phase in ('train', 'valid', 'test'):
        try:
            with open(os.path.join(data_dir, 'vgdb_2016',
                                   '%s.pickle' % phase), 'rb') as f:
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

        data['train'] = X[train_indices], y[train_indices], names[
            train_indices]
        data['valid'] = X[valid_indices], y[valid_indices], names[
            valid_indices]

    print('train-valid split:', data['train'][0].shape[0],
          data['valid'][0].shape[0])

    train_data = triplets_gen(*data['train'], embedding_net=embedding_net, batch_size=batch_size)
    valid_data = triplets_gen(*data['valid'], embedding_net=embedding_net, batch_size=batch_size)

    print('training network...')
    try:
        training_net.fit_generator(
            train_data, samples_per_epoch=train_samples_per_epoch,
            validation_data=valid_data, nb_val_samples=nb_val_samples,
            nb_epoch=nb_epoch, nb_worker=nb_worker,
            callbacks=[
                callbacks.ReduceLROnPlateau(patience=reduce_lr_on_plateau_patience),
                callbacks.TensorBoard(tensorboard_file, write_graph=False),
                callbacks.ModelCheckpoint(ckpt_file, verbose=1, save_best_only=True),
            ])
    except KeyboardInterrupt:
        print('training interrupted by user.')
