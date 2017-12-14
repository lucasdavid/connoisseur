"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import tensorflow as tf
from PIL import ImageFile
from keras import optimizers, backend as K
from keras.callbacks import TerminateOnNaN, EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment, utils as sacred_utils

from connoisseur import get_preprocess_fn
from connoisseur.datasets.painter_by_numbers import load_multiple_outputs
from connoisseur.models import build_model
from connoisseur.utils.image import MultipleOutputsDirectorySequence

ex = Experiment('train-network-multiple-predictions')

ex.captured_out_filter = sacred_utils.apply_backspaces_and_linefeeds
ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    data_dir = "/datasets/pbn/random_299/"
    batch_size = 64
    image_shape = [299, 299, 3]
    train_shuffle = True
    valid_shuffle = False
    train_info = '/datasets/pbn/train_info.csv'
    subdirectories = None

    architecture = 'InceptionV3'
    tag = 'pbn_%s_multilabel' % architecture.lower()
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    ckpt_file = '%s.hdf5' % tag

    device = "/gpu:0"

    opt_params = {'lr': .001}
    dropout_p = 0.2
    resuming_from = None
    epochs = 500
    steps_per_epoch = None
    workers = 8
    use_multiprocessing = False
    initial_epoch = 0
    early_stop_patience = 20
    tensorboard_tag = 'train-multiple-outputs-network-%s' % tag
    first_trainable_layer = None
    outputs_meta = [
        {'n': 'artist', 'u': 1584, 'a': 'softmax', 'l': 'categorical_crossentropy', 'm': 'accuracy', 'w': .4},
        {'n': 'style', 'u': 135, 'a': 'softmax', 'l': 'categorical_crossentropy', 'm': 'accuracy', 'w': .3},
        {'n': 'genre', 'u': 42, 'a': 'softmax', 'l': 'categorical_crossentropy', 'm': 'accuracy', 'w': .3},
        # {'n': 'date', 'u': 1, 'a': 'linear', 'l': 'mse', 'm': 'mse', 'w': .1}
    ]


@ex.automain
def run(_run, data_dir, subdirectories, train_shuffle, valid_shuffle, image_shape, train_info, batch_size,
        architecture, weights, last_base_layer, use_gram_matrix, pooling, dropout_p, device,
        epochs, steps_per_epoch, initial_epoch, opt_params, resuming_from, ckpt_file,
        workers, use_multiprocessing, early_stop_patience, first_trainable_layer, tensorboard_tag,
        outputs_meta):
    try:
        report_dir = _run.observers[0].dir
    except IndexError:
        report_dir = './logs/_unlabeled'

    print('reading train-info...')
    outputs, name_map = load_multiple_outputs(train_info, outputs_meta)

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=.2,
        rotation_range=.2,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=get_preprocess_fn(architecture))

    train_data = MultipleOutputsDirectorySequence(os.path.join(data_dir, 'train'), outputs, name_map, g,
                                                  batch_size=batch_size, target_size=image_shape[:2],
                                                  subdirectories=subdirectories,
                                                  shuffle=train_shuffle)
    valid_data = MultipleOutputsDirectorySequence(os.path.join(data_dir, 'valid'), outputs, name_map, g,
                                                  batch_size=batch_size, target_size=image_shape[:2],
                                                  subdirectories=subdirectories,
                                                  shuffle=valid_shuffle)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture,
                            weights=weights, dropout_p=dropout_p,
                            last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            include_top=True,
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
            model.fit_generator(train_data,
                                steps_per_epoch=steps_per_epoch,
                                epochs=epochs,
                                validation_data=valid_data,
                                initial_epoch=initial_epoch, verbose=1,
                                workers=workers, use_multiprocessing=use_multiprocessing,
                                callbacks=[
                                    TerminateOnNaN(),
                                    EarlyStopping(patience=early_stop_patience),
                                    ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                                    TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                                    ModelCheckpoint(os.path.join(report_dir, ckpt_file), save_best_only=True, verbose=1)
                                ])

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
