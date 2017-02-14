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

from base import combine_pairs_for_training_gen, evaluate
from model import build_model

ex = Experiment('3-train-contrastive')


@ex.config
def config():
    batch_size = 64
    device = "/gpu:0"
    data_dir = "/datasets/ldavid/van_gogh"

    opt_params = {'lr': 0.0005, 'decay': 1e-4}
    ckpt_file = './ckpt/opt-weights.hdf5'
    nb_epoch = 200
    train_samples_per_epoch = 2048
    nb_val_samples = 260
    nb_worker = 1
    early_stop_patience = 10
    tensorboard_file = './logs/contrastive-loss-3-fc'


@ex.automain
def run(batch_size, data_dir,
        device, opt_params, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, tensorboard_file):
    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    K.set_session(sess)

    data = {}
    for phase in ('train', 'test'):
        with open(os.path.join(data_dir, 'vgdb_2016', '%s.pickle' % phase), 'rb') as f:
            data[phase] = pickle.load(f)
            # Data as-it-is and categorical labels.
            data[phase] = (data[phase]['data'],
                           np.argmax(data[phase]['target'], axis=-1),
                           np.array(data[phase]['names'], copy=False))

    # Separate train and valid sets.
    s = np.arange(data['train'][0].shape[0])
    np.random.shuffle(s)
    X, y, names = data['train']
    for phase, samples in zip(('train', 'valid'),
                              (s[nb_val_samples:], s[:nb_val_samples])):
        X, y, names = X[s], y[s], names[s]
        data[phase] = X, y, names

    train_data = combine_pairs_for_training_gen(*data['train'], batch_size=batch_size)
    valid_data = combine_pairs_for_training_gen(*data['valid'], batch_size=batch_size)

    model = build_model(x_shape=(2048,), opt_params=opt_params, device=device, compile_opt=True)

    print('training network...')
    try:
        model.fit_generator(
            train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
            validation_data=valid_data, nb_val_samples=nb_val_samples, nb_worker=nb_worker,
            callbacks=[
                callbacks.EarlyStopping(patience=early_stop_patience),
                callbacks.TensorBoard(tensorboard_file, write_graph=False),
                callbacks.ModelCheckpoint(ckpt_file, verbose=1, save_best_only=True),
            ])
    except KeyboardInterrupt:
        print('training interrupted by user.')
