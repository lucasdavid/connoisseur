"""Transfer and fine-tune InceptionV3 on PainterByNumbers dataset.

Uses InceptionV3 trained over `imagenet` and fine-tune it to PainterByNumbers dataset.
Image patches are loaded directly from the disk. Finally, train an SVM over
the fine-tuned extraction network.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os

import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from model import build_model
from sacred import Experiment

from connoisseur import utils

ex = Experiment('4-1-train-inception')


@ex.config
def config():
    batch_size = 64
    image_shape = [299, 299, 3]
    train_shuffle = True
    train_augmentations = []
    dataset_train_seed = 12
    valid_shuffle = True
    valid_augmentations = []
    dataset_valid_seed = 98
    device = "/gpu:0"
    data_dir = "/datasets/painter-by-numbers"
    augmentation_variability = .1

    inception_optimal_params = {'lr': 7.4e-05}
    dropout_p = 0.5
    ckpt_file = './ckpt/inception{epoch:02d}-{val_loss:.2f}.hdf5'
    nb_epoch = 500
    train_samples_per_epoch = 71490
    nb_val_samples = 3168
    nb_worker = 8
    early_stop_patience = 30
    tensorboard_file = './logs/inception,lr:7.4e-05,weights:imagenet,augmentation:.1'


@ex.automain
def run(image_shape, batch_size, data_dir,
        train_shuffle, train_augmentations, dataset_train_seed,
        valid_shuffle, valid_augmentations, dataset_valid_seed,
        augmentation_variability,

        device, inception_optimal_params, dropout_p, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker,
        early_stop_patience, tensorboard_file):
    tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    # These parameters produce the same results of `preprocess_input`.
    g = utils.image.ImageDataGenerator(
        featurewise_center=True,
        rescale=2. / 255.,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=180,
        zoom_range=.2,
        width_shift_range=.2,
        height_shift_range=.2,
        fill_mode='reflect')
    g.mean = 1

    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=image_shape[:2], extraction_method='random-crop',
        augmentations=train_augmentations, augmentation_variability=augmentation_variability,
        batch_size=batch_size, shuffle=train_shuffle, seed=dataset_train_seed)

    val_data = g.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=image_shape[:2], extraction_method='random-crop',
        augmentations=valid_augmentations, augmentation_variability=augmentation_variability,
        batch_size=batch_size, shuffle=valid_shuffle, seed=dataset_valid_seed)

    labels = os.listdir(os.path.join(data_dir, 'train'))

    try:
        with tf.device(device):
            print('building model...')
            images = Input(batch_shape=[None] + image_shape)
            base_model = InceptionV3(weights='imagenet', input_tensor=images, include_top=False)
            x = base_model.output
            x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
            x = Flatten(name='flatten')(x)
            x = Dense(len(labels), activation='softmax', name='predictions')(x)

            model = Model(input=base_model.input, output=x)

            opt = optimizers.Adadelta()
            model.compile(optimizer=opt, metrics=['accuracy'], loss='categorical_crossentropy')

            print('training InceptionV3...')
            model.fit_generator(
                generator=train_data, samples_per_epoch=train_samples_per_epoch, nb_epoch=nb_epoch,
                validation_data=val_data, nb_val_samples=nb_val_samples,
                nb_worker=nb_worker, verbose=1,
                callbacks=[
                    callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * inception_optimal_params['lr']),
                    callbacks.ReduceLROnPlateau(min_lr=.0000001, patience=50),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file, write_graph=False),
                    callbacks.ModelCheckpoint(ckpt_file, period=20, verbose=1),
                ])
            print('training completed.')
    except KeyboardInterrupt:
        print('training interrupted by user.')
