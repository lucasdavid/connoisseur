"""Transfer and fine-tune models on PainterByNumbers dataset.

Uses an architecture to train over PainterByNumbers dataset. Image patches are
loaded directly from the disk. Finally, train an SVM over the fine-tuned
extraction network.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os

import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from sacred import Experiment

from connoisseur import utils
from connoisseur.models.painter_by_numbers import build_model

ex = Experiment('painter-by-numbers.4-train.train-network')


@ex.config
def config():
    data_dir = "/datasets/painter-by-numbers"
    batch_size = 64
    image_shape = [299, 299, 3]
    architecture = 'inception'
    weights = 'imagenet'
    train_shuffle = True
    train_augmentations = []
    dataset_train_seed = 12
    valid_shuffle = True
    valid_augmentations = []
    dataset_valid_seed = 98
    device = "/gpu:0"
    augmentation_variability = .01

    opt_params = {'lr': .001}
    dropout_p = 0.5
    resuming = False
    ckpt_file = './ckpt/opt-weights.hdf5'
    nb_epoch = 500
    train_samples_per_epoch = 12672
    nb_val_samples = 3168
    nb_worker = 8
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_file = './logs/%s,%s,batch-size:%i' % (architecture, opt_params, batch_size)


@ex.automain
def run(image_shape, architecture, weights, batch_size, data_dir,
        train_shuffle, train_augmentations, dataset_train_seed,
        valid_shuffle, valid_augmentations, dataset_valid_seed,
        augmentation_variability,

        device, opt_params, dropout_p, resuming, ckpt_file,
        train_samples_per_epoch, nb_epoch,
        nb_val_samples, nb_worker, initial_epoch,
        early_stop_patience, tensorboard_file):
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    g = utils.image.ImageDataGenerator(
        featurewise_center=True,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        height_shift_range=.2,
        width_shift_range=.2,
        rescale=2. / 255.,
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

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, arch=architecture, weights=weights, dropout_p=dropout_p)
        model.compile(optimizer=optimizers.Adam(**opt_params),
                      metrics=['accuracy'],
                      loss='categorical_crossentropy')

        if resuming:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data, train_samples_per_epoch, nb_epoch, verbose=1,
                validation_data=val_data, nb_val_samples=nb_val_samples,
                nb_worker=nb_worker,
                callbacks=[
                    callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * opt_params['lr']),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 2)),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file, histogram_freq=5),
                    callbacks.ModelCheckpoint(ckpt_file, save_best_only=True, verbose=1),
                ],
                initial_epoch=initial_epoch)

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
