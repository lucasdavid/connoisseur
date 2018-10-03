"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras import optimizers, backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sacred import Experiment, utils as sacred_utils

from connoisseur.datasets import load_pickle_data
from connoisseur.datasets.painter_by_numbers import load_multiple_outputs
from connoisseur.models import build_meta_limb

ex = Experiment('train-meta-network-multiple-predictions')

ex.captured_out_filter = sacred_utils.apply_backspaces_and_linefeeds
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    data_dir = "/datasets/pbn/random_299/"
    batch_size = 4096
    shape = [1536]
    device = "/gpu:0"
    train_info = '/datasets/pbn/train_info.csv'

    use_gram_matrix = False
    ckpt_file = 'weights.hdf5'

    opt_params = {'lr': .001}
    dropout_p = 0.2
    resuming_from = None
    epochs = 500
    steps_per_epoch = None
    validation_steps = None
    initial_epoch = 0
    early_stop_patience = 20
    first_trainable_layer = None
    class_weight = None
    outputs_meta = [
        {'n': 'artist', 'u': 1584, 'a': 'softmax', 'l': 'sparse_categorical_crossentropy', 'm': 'accuracy', 'w': .6},
        {'n': 'style', 'u': 135, 'a': 'softmax', 'l': 'sparse_categorical_crossentropy', 'm': 'accuracy', 'w': .2},
        {'n': 'genre', 'u': 42, 'a': 'softmax', 'l': 'sparse_categorical_crossentropy', 'm': 'accuracy', 'w': .2},
        # {'n': 'date', 'u': 1, 'a': 'linear', 'l': 'mse', 'm': 'mse', 'w': .1}
    ]
    dense_layers = []
    layer_name = 'global_average_pooling2d_1'
    chunks = (0, 1, 2, 3, 4)


@ex.automain
def run(_run, data_dir, shape, batch_size, device, train_info,
        use_gram_matrix, ckpt_file, dense_layers,
        opt_params, dropout_p, resuming_from,
        epochs, steps_per_epoch, validation_steps,
        initial_epoch, early_stop_patience, first_trainable_layer,
        class_weight, outputs_meta, layer_name, chunks):
    try:
        report_dir = _run.observers[0].dir
    except IndexError:
        report_dir = './logs/_unlabeled'

    print('loading limb-embedded inputs...')
    d = load_pickle_data(data_dir,
                         keys=['data', 'names'],
                         phases=['train', 'valid'],
                         chunks=chunks)
    (x_train, names_train), (x_valid, names_valid) = d['train'], d['valid']
    x_train, x_valid = (x[layer_name] for x in (x_train, x_valid))
    print('x-train, x-valid shape:', x_train.shape, x_valid.shape)

    p = np.arange(len(x_train))
    np.random.shuffle(p)
    x_train = x_train[p]
    names_train = names_train[p]

    p = np.arange(len(x_valid))
    np.random.shuffle(p)
    x_valid = x_valid[p]
    names_valid = names_valid[p]

    print('loading labels...')
    outputs, name_map = load_multiple_outputs(train_info, outputs_meta, encode='sparse')

    ys = []
    for phase, names in zip(('train', 'valid'),
                            (names_train, names_valid)):
        names = ['-'.join(os.path.basename(n).split('-')[:-1]) for n in names]
        indices = [name_map[n] for n in names]
        ys += [{o: v[indices] for o, v in outputs.items()}]

    y_train, y_valid = ys

    # class_weight = sk_utils.compute_class_weight(class_weight, np.unique(y_train), y_train)
    print('data sample:')
    print(x_train[:10])

    print(x_train.shape)
    print(y_train['artist'].shape)

    with tf.device(device):
        print('building...')
        model = build_meta_limb(shape, dropout_p=dropout_p,
                                use_gram_matrix=use_gram_matrix,
                                include_top=True,
                                dense_layers=dense_layers,
                                classes=[o['u'] for o in outputs_meta],
                                predictions_name=[o['n'] for o in outputs_meta],
                                predictions_activation=[o['a'] for o in outputs_meta])
        layer_names = [l.name for l in model.layers]
        if first_trainable_layer:
            if first_trainable_layer not in layer_names:
                raise ValueError('%s is not a layer in the model: %s'
                                 % (first_trainable_layer, layer_names))
            _trainable = False
            for layer in model.layers:
                if layer.name == first_trainable_layer:
                    _trainable = True
                layer.trainable = _trainable
            del _trainable
        model.compile(optimizer=optimizers.Adam(**opt_params),
                      loss=dict((o['n'], o['l']) for o in outputs_meta),
                      metrics=dict((o['n'], o['m']) for o in outputs_meta),
                      loss_weights=dict((o['n'], o['w']) for o in outputs_meta))
        model.summary()

        if resuming_from:
            print('re-loading weights...')
            model.load_weights(resuming_from)

        try:
            print('training from epoch %i...' % initial_epoch)
            model.fit(x_train, y_train,
                      epochs=epochs,
                      batch_size=batch_size,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=(x_valid, y_valid),
                      validation_steps=validation_steps,
                      initial_epoch=initial_epoch,
                      verbose=2,
                      class_weight=class_weight,
                      callbacks=[
                          TerminateOnNaN(),
                          EarlyStopping(patience=early_stop_patience),
                          ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                          # TensorBoard(report_dir,
                          #             batch_size=batch_size, write_grads=True, write_images=True,
                          #             histogram_freq=10),
                          ModelCheckpoint(os.path.join(report_dir, ckpt_file), save_best_only=True, verbose=1)
                      ])
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
        finally:
            print('train history:', model.history.history)
