"""Train a network on top of the network trained on Painters-by-numbers.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from connoisseur.datasets import load_pickle_data
from connoisseur.datasets.painter_by_numbers import load_multiple_outputs
from connoisseur.utils.image import create_pairs
from connoisseur.models import build_siamese_meta

ImageFile.LOAD_TRUNCATED_IMAGES = True

ex = Experiment('train-top-meta-network-mo')

ex.captured_out_filter = apply_backspaces_and_linefeeds
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)
K.set_session(session)


@ex.config
def config():
    device = "/gpu:0"

    train_info = '/datasets/pbn/train_info.csv'
    data_dir = "/datasets/pbn/random_299/"
    chunks = (0, 1, 2)
    train_pairs = 1584
    valid_pairs = 1584
    train_shuffle = True
    valid_shuffle = True

    batch_size = 128
    opt_params = {'lr': .001}
    dropout_rate = .5
    ckpt = 'weights.h5'
    resuming_ckpt = None
    epochs = 100
    steps_per_epoch = None
    validation_steps = None
    initial_epoch = 0
    early_stop_patience = 30

    outputs_meta = [
        dict(n='artist', u=1584, e=1024, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='style', u=135, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='genre', u=42, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='date', u=1, e=256, j='l2', a='linear', l='mse', m='mae')
    ]

    verbose = 1


@ex.automain
def run(_run, train_info, data_dir, chunks, train_pairs, valid_pairs, train_shuffle, valid_shuffle,
        batch_size, device,
        opt_params, dropout_rate, resuming_ckpt, ckpt, steps_per_epoch, epochs,
        validation_steps, initial_epoch, early_stop_patience,
        outputs_meta, verbose):
    report_dir = _run.observers[0].dir

    print('loading limb-embedded inputs...')
    d = load_pickle_data(data_dir,
                         keys=['data', 'names'],
                         phases=['train', 'valid'],
                         chunks=chunks)
    x_train, x_valid = d['train'][0], d['valid'][0]
    print('x-train, x-valid shape:', x_train['artist'].shape, x_valid['artist'].shape)

    print('loading labels...')
    outputs, name_map = load_multiple_outputs(train_info, outputs_meta, encode='sparse')

    ys = []
    for phase in ('train', 'valid'):
        names = d[phase][1]
        names = ['-'.join(os.path.basename(n).split('-')[:-1]) for n in names]
        indices = [name_map[n] for n in names]
        ys += [{o: v[indices] for o, v in outputs.items()}]

    y_train, y_valid = ys

    artists = np.unique(y_train['artist'])
    x_train, y_train = create_pairs(x_train, y_train,
                                    pairs=train_pairs,
                                    classes=artists,
                                    shuffle=train_shuffle)

    x_valid, y_valid = create_pairs(x_valid, y_valid,
                                    pairs=valid_pairs,
                                    classes=artists,
                                    shuffle=valid_shuffle)

    with tf.device(device):
        print('building...')
        model = build_siamese_meta(outputs_meta, dropout_rate=dropout_rate)

        print('siamese model summary:')
        model.summary()
        if resuming_ckpt:
            print('loading weights...')
            model.load_weights(resuming_ckpt)

        model.compile(optimizer=optimizers.Adam(**opt_params),
                      loss=dict((o['n'] + '_binary_predictions', o['l']) for o in outputs_meta),
                      metrics=dict((o['n'] + '_binary_predictions', o['m']) for o in outputs_meta))

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit(x_train, y_train,
                      steps_per_epoch=steps_per_epoch,
                      epochs=epochs,
                      validation_data=(x_valid, y_valid),
                      validation_steps=validation_steps,
                      initial_epoch=initial_epoch,
                      verbose=verbose,
                      callbacks=[
                          callbacks.TerminateOnNaN(),
                          callbacks.EarlyStopping(patience=early_stop_patience),
                          callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=early_stop_patience // 3),
                          callbacks.TensorBoard(report_dir,
                                                batch_size=batch_size,
                                                histogram_freq=1, write_grads=True, write_images=True),
                          callbacks.ModelCheckpoint(os.path.join(report_dir, ckpt), save_best_only=True, verbose=1),
                      ])
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
