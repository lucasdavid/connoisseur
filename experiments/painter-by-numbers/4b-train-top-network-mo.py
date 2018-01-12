"""Train a network on top of the network trained on Painters-by-numbers.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from connoisseur import get_preprocess_fn
from connoisseur.datasets.painter_by_numbers import load_multiple_outputs
from connoisseur.models import build_siamese_model
from connoisseur.utils.image import BalancedDirectoryPairsMultipleOutputsSequence


ImageFile.LOAD_TRUNCATED_IMAGES = True

ex = Experiment('train-top-network-multiple-outputs')

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
    subdirectories = None
    train_pairs = 1584
    valid_pairs = 1584
    train_shuffle = True
    valid_shuffle = True

    batch_size = 128
    image_shape = [299, 299, 3]
    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = ()
    trainable_limbs = False
    pooling = 'avg'

    limb_weights = '/work/painter-by-numbers/wlogs/train-multiple-outputs/5/weights.hdf5'

    opt_params = {'lr': .001}
    dropout_rate = .5
    ckpt = 'weights.hdf5'
    resuming_ckpt = None
    epochs = 100
    steps_per_epoch = None
    validation_steps = None
    use_multiprocessing = False
    workers = 1
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_tag = 'training/'

    outputs_meta = [
        dict(n='artist', u=1584, e=1024, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='style', u=135, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='genre', u=42, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        # dict(n='date', u=1, e=256, j='l2', a='linear', l=utils.contrastive_loss, m=utils.contrastive_accuracy)
    ]


@ex.automain
def run(_run, image_shape, train_info, data_dir, train_pairs, valid_pairs, train_shuffle, valid_shuffle,
        subdirectories, architecture, weights, batch_size, last_base_layer, pooling, device,
        opt_params, dropout_rate, resuming_ckpt, ckpt, steps_per_epoch, epochs,
        validation_steps, workers, use_multiprocessing, initial_epoch, early_stop_patience, use_gram_matrix,
        dense_layers,
        limb_weights, trainable_limbs, tensorboard_tag, outputs_meta):
    report_dir = _run.observers[0].dir

    print('reading train-info...')
    outputs, name_map = load_multiple_outputs(train_info, outputs_meta, encode='sparse')

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=.2,
        rotation_range=.2,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=get_preprocess_fn(architecture))

    print('loading train meta-data...')
    train_data = BalancedDirectoryPairsMultipleOutputsSequence(
        os.path.join(data_dir, 'train'), outputs, name_map, g,
        batch_size=batch_size, target_size=image_shape[:2],
        subdirectories=subdirectories,
        shuffle=train_shuffle,
        pairs=train_pairs)

    print('loading valid meta-data...')
    valid_data = BalancedDirectoryPairsMultipleOutputsSequence(
        os.path.join(data_dir, 'valid'), outputs, name_map, g,
        batch_size=batch_size, target_size=image_shape[:2],
        subdirectories=subdirectories,
        shuffle=valid_shuffle,
        pairs=valid_pairs)

    with tf.device(device):
        print('building...')
        model = build_siamese_model(image_shape, architecture, dropout_rate, weights,
                                    last_base_layer=last_base_layer,
                                    use_gram_matrix=use_gram_matrix, dense_layers=dense_layers, pooling=pooling,
                                    include_base_top=False, include_top=True,
                                    trainable_limbs=trainable_limbs,
                                    limb_weights=limb_weights,
                                    predictions_activation=[o['a'] for o in outputs_meta],
                                    predictions_name=[o['n'] for o in outputs_meta],
                                    classes=[o['u'] for o in outputs_meta],
                                    embedding_units=[o['e'] for o in outputs_meta],
                                    joints=[o['j'] for o in outputs_meta])

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
            model.fit_generator(
                train_data,
                steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=valid_data,
                validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                use_multiprocessing=use_multiprocessing, workers=workers, verbose=2,
                callbacks=[
                    callbacks.TerminateOnNaN(),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=early_stop_patience // 3),
                    callbacks.TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                    callbacks.ModelCheckpoint(os.path.join(report_dir, ckpt), save_best_only=True, verbose=1),
                ])
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
