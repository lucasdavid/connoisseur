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

from connoisseur import utils
from connoisseur.models import build_siamese_model
from connoisseur.utils.image import BalancedDirectoryPairsSequence

ImageFile.LOAD_TRUNCATED_IMAGES = True

ex = Experiment('train-top-network')

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
    train_pairs = 1584
    valid_pairs = 1584
    num_classes = 1584
    classes = None

    batch_size = 128
    image_shape = [299, 299, 3]
    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = ()
    embedding_units = 1024
    joints = 'multiply'
    trainable_limbs = False
    pooling = 'avg'

    predictions_activation = 'softmax'
    limb_weights = '/work/painter-by-numbers/ckpt/limb_weights.hdf5'

    opt_params = {'lr': .001}
    dropout_rate = 0.2
    ckpt = 'top-network.hdf5'
    resuming_ckpt = None
    epochs = 100
    steps_per_epoch = None
    validation_steps = None
    use_multiprocessing = False
    workers = 1
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_tag = 'train-top-network/'


@ex.automain
def run(_run, image_shape, data_dir, train_pairs, valid_pairs, classes,
        num_classes, architecture, weights, batch_size, last_base_layer, pooling, device, predictions_activation,
        opt_params, dropout_rate, resuming_ckpt, ckpt, steps_per_epoch, epochs, validation_steps, joints,
        workers, use_multiprocessing, initial_epoch, early_stop_patience, use_gram_matrix, dense_layers,
        embedding_units, limb_weights, trainable_limbs, tensorboard_tag):
    report_dir = _run.observers[0].dir

    if isinstance(classes, int):
        classes = sorted(os.listdir(os.path.join(data_dir, 'train')))[:classes]

    g = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=.2, rotation_range=.2,
                           height_shift_range=.2, width_shift_range=.2,
                           fill_mode='reflect', preprocessing_function=utils.get_preprocess_fn(architecture))

    train_data = BalancedDirectoryPairsSequence(os.path.join(data_dir, 'train'), g, target_size=image_shape[:2],
                                                pairs=train_pairs, classes=classes, batch_size=batch_size)
    valid_data = BalancedDirectoryPairsSequence(os.path.join(data_dir, 'valid'), g, target_size=image_shape[:2],
                                                pairs=valid_pairs, classes=classes, batch_size=batch_size)
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(valid_data)

    with tf.device(device):
        print('building...')
        model = build_siamese_model(image_shape, architecture, dropout_rate, weights, num_classes, last_base_layer,
                                    use_gram_matrix, dense_layers, pooling, include_base_top=False, include_top=True,
                                    predictions_activation=predictions_activation, limb_weights=limb_weights,
                                    trainable_limbs=trainable_limbs, embedding_units=embedding_units, joints=joints)
        print('siamese model summary:')
        model.summary()
        if resuming_ckpt:
            print('loading weights...')
            model.load_weights(resuming_ckpt)

        model.compile(loss='binary_crossentropy',
                      metrics=['accuracy'],
                      optimizer=optimizers.Adam(**opt_params))

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data,
                steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=valid_data,
                validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                use_multiprocessing=use_multiprocessing, workers=workers, verbose=1,
                callbacks=[
                    callbacks.TerminateOnNaN(),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                    callbacks.ModelCheckpoint(os.path.join(report_dir, ckpt), save_best_only=True, save_weights_only=True, verbose=1),
                ])
        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
