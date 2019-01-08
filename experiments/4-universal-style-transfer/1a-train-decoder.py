import os

import numpy as np
import tensorflow as tf
from keras import backend as K, applications, Model, utils, callbacks, optimizers, losses
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment

from connoisseur.utils import get_preprocess_fn, gram_matrix

ex = Experiment('train-gram-network')

tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    data_dir = '/datasets/pbn/random_299'
    batch_size = 64
    image_shape = [300, 300, 3]
    architecture = 'VGG16'
    weights = None
    base_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3']
    use_gram_matrix = True
    dense_layers = ()
    pooling = 'avg'
    train_shuffle = True
    train_seed = 12
    valid_shuffle = False
    validation_seed = 98
    device = '/cpu:0'

    classes = None
    frozen_base = True

    opt_params = {'lr': .001}
    dropout_p = 0.4
    resuming_from_ckpt_file = None
    epochs = 2
    steps_per_epoch = 2
    validation_steps = 2
    workers = 8
    use_multiprocessing = False
    initial_epoch = 0
    early_stop_patience = 30
    class_weight = 'balanced'
    class_mode = 'input'


@ex.capture
def build_model(_run, architecture, weights, image_shape, base_layers, frozen_base):
    report_dir = _run.observers[0].dir if _run.observers else './'

    commons = dict(kernel_size=(3, 3), padding='same', activation='relu')

    enc_cls = getattr(applications, architecture)
    encoder = enc_cls(include_top=False,
                      weights=weights,
                      input_shape=image_shape)

    inversions = [
        Conv2DTranspose(3, **commons),
        Conv2DTranspose(64, **commons),
        UpSampling2D((2, 2)),
        Conv2DTranspose(64, **commons),
        Conv2DTranspose(128, **commons),
        UpSampling2D((2, 2)),
        Conv2DTranspose(128, **commons),
        Conv2DTranspose(256, **commons),
        Conv2DTranspose(256, **commons),
        UpSampling2D((2, 2)),
        Conv2DTranspose(256, **commons),
        Conv2DTranspose(512, **commons),
        Conv2DTranspose(512, **commons),
        UpSampling2D((2, 2)),
        Conv2DTranspose(512, **commons),
        Conv2DTranspose(512, **commons),
        Conv2DTranspose(512, **commons),
        UpSampling2D((2, 2)),
    ]

    decoder_outputs = []

    for name in base_layers:
        print(f'Building {name} decoder')

        bl = encoder.get_layer(name)
        at = encoder.layers.index(bl)
        y = bl.output

        layer_inversions = inversions[:at][::-1]

        for l in layer_inversions:
            y = l(y)

        decoder_outputs.append(y)

    decoder = Model(inputs=encoder.inputs, outputs=decoder_outputs)

    if frozen_base:
        for l in decoder.layers:
            if l.name.startswith('block'):
                l.trainable = False

    print('Encoder:')
    encoder.summary()
    print('Decoder:')
    decoder.summary()
    utils.plot_model(decoder, os.path.join(report_dir, 'model.png'))

    return encoder, decoder


def content_style_loss(y, p, a=.2):
    return a * losses.mse(y, p) + (1 - a) * losses.mse(gram_matrix(y), gram_matrix(p))


def multiple_target_generator(gen, repeat=3):
    for (x, y) in gen:
        yield x, repeat * [y]


@ex.automain
def run(_run, image_shape, data_dir, train_shuffle, train_seed, valid_shuffle, validation_seed,
        class_mode, architecture, batch_size, device, opt_params, steps_per_epoch,
        epochs, validation_steps, workers, use_multiprocessing, initial_epoch, early_stop_patience):
    report_dir = _run.observers[0].dir if _run.observers else './'

    g = ImageDataGenerator(preprocessing_function=get_preprocess_fn(architecture), validation_split=0.3)
    training = g.flow_from_directory(os.path.join(data_dir, 'train'),
                                     target_size=image_shape[:2],
                                     class_mode=class_mode,
                                     shuffle=train_shuffle,
                                     subset='training',
                                     seed=train_seed)
    validation = g.flow_from_directory(os.path.join(data_dir, 'train'),
                                       target_size=image_shape[:2],
                                       class_mode=class_mode,
                                       shuffle=valid_shuffle,
                                       subset='validation',
                                       seed=validation_seed)

    with tf.device(device):
        encoder, decoder = build_model()

        decoder.compile(optimizer=optimizers.Adam(**opt_params),
                        loss=content_style_loss)

        decoder.fit_generator(multiple_target_generator(training),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              verbose=1,
                              validation_data=multiple_target_generator(validation),
                              validation_steps=validation_steps,
                              use_multiprocessing=use_multiprocessing,
                              initial_epoch=initial_epoch,
                              workers=workers,
                              callbacks=[
                                  callbacks.EarlyStopping(patience=early_stop_patience),
                                  callbacks.ReduceLROnPlateau(patience=early_stop_patience // 3),
                                  callbacks.TensorBoard(report_dir, batch_size=batch_size),
                                  callbacks.ModelCheckpoint(os.path.join(report_dir, 'decoder.h5'), save_best_only=True)
                              ])
