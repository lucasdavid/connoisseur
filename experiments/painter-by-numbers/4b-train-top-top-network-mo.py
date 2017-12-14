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
from connoisseur.models import build_siamese_mo_model
from connoisseur.utils.image import BalancedDirectoryPairsSequence

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

    data_dir = "/datasets/pbn/random_299/"
    classes = None
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
    limb_dense_layers = ()
    trainable_limbs = False
    trainable_joints = False
    pooling = 'avg'
    dense_layers = (128, 128)

    limb_weights = '/work/painter-by-numbers/wlogs/train-multiple-outputs/5/weights.hdf5'
    joint_weights = '/work/painter-by-numbers/wlogs/train-top-mo/5/weights.hdf5'

    opt_params = {'lr': .001}
    dropout_rate = .5
    ckpt = 'weights.hdf5'
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
        dict(n='style', u=135, e=256, j='multiply', a='softmax',
             l='binary_crossentropy', m='accuracy'),
        dict(n='genre', u=42, e=256, j='multiply', a='softmax',
             l='binary_crossentropy', m='accuracy'),
        # dict(n='date', u=1, e=256, j='l2', a='linear', l=utils.contrastive_loss, m=utils.contrastive_accuracy)
    ]


@ex.automain
def run(_run, image_shape, data_dir,
        train_pairs, valid_pairs, train_shuffle, valid_shuffle,
        classes, architecture, weights, batch_size, last_base_layer, pooling,
        device, opt_params, dropout_rate, ckpt, steps_per_epoch, epochs,
        validation_steps, workers, use_multiprocessing, initial_epoch,
        early_stop_patience, use_gram_matrix, limb_dense_layers,
        limb_weights, trainable_limbs, joint_weights, trainable_joints,
        dense_layers, tensorboard_tag, outputs_meta):
    report_dir = _run.observers[0].dir

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
    train_data = BalancedDirectoryPairsSequence(
        os.path.join(data_dir, 'train'), g,
        batch_size=batch_size, target_size=image_shape[:2],
        classes=classes,
        shuffle=train_shuffle,
        pairs=train_pairs)

    print('loading valid meta-data...')
    valid_data = BalancedDirectoryPairsSequence(
        os.path.join(data_dir, 'valid'), g,
        batch_size=batch_size, target_size=image_shape[:2],
        classes=classes,
        shuffle=valid_shuffle,
        pairs=valid_pairs)

    with tf.device(device):
        print('building...')
        model = build_siamese_mo_model(
            image_shape, architecture, outputs_meta,
            dropout_rate, weights,
            last_base_layer=last_base_layer,
            use_gram_matrix=use_gram_matrix,
            limb_dense_layers=limb_dense_layers,
            pooling=pooling,
            trainable_limbs=trainable_limbs,
            limb_weights=limb_weights,
            trainable_joints=trainable_joints,
            joint_weights=joint_weights,
            dense_layers=dense_layers)

        model.compile(optimizer=optimizers.Adam(**opt_params),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data,
                steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=valid_data,
                validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                use_multiprocessing=use_multiprocessing, workers=workers,
                verbose=1,
                callbacks=[
                    callbacks.TerminateOnNaN(),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                    callbacks.ModelCheckpoint(os.path.join(report_dir, ckpt), save_best_only=True, verbose=1),
                ])
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
