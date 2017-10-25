"""1 Train Siamese Gram Network

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import numpy as np
from sacred import Experiment
from PIL import Image, ImageFile
import tensorflow as tf
from keras import Input, layers, optimizers, callbacks, backend as K
from keras.engine import Model
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from connoisseur.utils.image import BalancedDirectoryPairsSequence
from connoisseur.utils import gram_matrix

ex = Experiment('1-train-siamese-gram-network')


@ex.config
def config():
    data_dir = '/media/files/datasets/vangogh/vgdb2016/'
    dataset_train_seed = 12
    dataset_valid_seed = 31
    batch_size = 32
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    n_classes = 2
    train_base_model = False
    ckpt_file = './checkpoints/m{%(id)s}.hdf5'

    device = "/cpu:0"
    opt_params = {'lr': .001}

    epochs = 500
    steps_per_epoch = 100
    validation_steps = 10
    workers = 8
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_file = './logs/1-train-network/vgg19,%s,batch-size:%i' % (opt_params, batch_size)


def euclidean(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def crop_central(image, sizes):
    width, height = image.size
    left = (width - sizes[1]) / 2
    top = (height - sizes[0]) / 2
    right = (width + sizes[1]) / 2
    bottom = (height + sizes[0]) / 2

    image.crop((left, top, right, bottom))
    return image


def get_input_sizes(data_dir, labels):
    # A lot of work here just to find the minimum width and height
    min_sizes = [np.inf, np.inf]
    max_sizes = [-np.inf, -np.inf]
    sum_sizes = [0, 0]
    n_samples = 0

    for p in ('train', 'valid'):
        for l in labels:
            for s in os.listdir(os.path.join(data_dir, p, l)):
                sizes = Image.open(os.path.join(data_dir, p, l, s)).size
                n_samples += 1

                for i in range(2):
                    if sizes[i] < min_sizes[i]:
                        min_sizes[i] = sizes[i]
                    if sizes[i] > max_sizes[i]:
                        max_sizes[i] = sizes[i]
                    sum_sizes[i] += sizes[i]
    # Let's use the averages.
    sizes = [int(x / n_samples) for x in sum_sizes]
    # Flipping PIL's format (W, H) to Keras's (H, W).
    return list(reversed(sizes))


@ex.automain
def main(data_dir, dataset_train_seed, dataset_valid_seed, batch_size, style_layers, n_classes,
         train_base_model, ckpt_file,
         device, opt_params,
         epochs, steps_per_epoch, validation_steps, workers, initial_epoch,
         early_stop_patience, tensorboard_file):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=.1,
        width_shift_range=.1,
        zoom_range=.2,
        fill_mode='reflect',
        preprocessing_function=preprocess_input)

    labels = sorted(os.listdir(os.path.join(data_dir, 'train')))[:n_classes]
    print('labels:', labels)

    sizes = get_input_sizes(data_dir, labels)
    input_shape = sizes + [3]
    print('input shape:', input_shape)

    with tf.device(device):
        # building siamese models.
        base_model = VGG19(input_shape=input_shape, include_top=False)
        base_model.trainable = train_base_model

        ia = Input(input_shape)
        ib = Input(input_shape)

        for m_id, l in enumerate(style_layers):
            y = base_model.get_layer(l).output
            # n_kernels = K.get_variable_shape(y)[-1]
            # y = layers.Lambda(gram_matrix, arguments=dict(norm_by_channels=False),
            #                   output_shape=[n_kernels, n_kernels])(y)
            y = layers.Flatten()(y)
            # y = Dense(1024, activation='relu')(y)
            # y = Dense(512, activation='relu')(y)
            m = Model(base_model.input, y)

            ya = m(ia)
            yb = m(ib)

            # y = layers.concatenate([ya, yb], axis=-1)
            y = layers.Lambda(euclidean, output_shape=lambda x: (x[0][0], 1))([ya, yb])
            y = layers.Dense(1, activation='sigmoid')(y)

            m = Model(inputs=[ia, ib], outputs=y)
            m.compile(optimizer=optimizers.Adam(**opt_params),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

            print('training model #%i from epoch %i' % (m_id, initial_epoch))

            train_data = BalancedDirectoryPairsSequence(
                directory=os.path.join(data_dir, 'train'),
                image_data_generator=g,
                target_size=input_shape[:2],
                classes=labels, batch_size=batch_size, shuffle=True,
                seed=dataset_train_seed)

            val_data = BalancedDirectoryPairsSequence(
                directory=os.path.join(data_dir, 'valid'),
                image_data_generator=g,
                target_size=input_shape[:2],
                classes=labels, batch_size=batch_size, shuffle=True,
                seed=dataset_valid_seed)

            try:
                m.fit_generator(
                    train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                    verbose=1, validation_data=val_data, workers=workers,
                    validation_steps=validation_steps,
                    callbacks=[
                        callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * opt_params['lr']),
                        callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                        callbacks.EarlyStopping(patience=early_stop_patience),
                        callbacks.TensorBoard(tensorboard_file),
                        callbacks.ModelCheckpoint(ckpt_file % {'id': m_id}, save_best_only=True, verbose=1),
                    ],
                    initial_epoch=initial_epoch)

            except KeyboardInterrupt:
                print('interrupted by user')
            else:
                print('done')
