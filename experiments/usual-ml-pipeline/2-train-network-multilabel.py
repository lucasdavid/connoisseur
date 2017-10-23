"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Note: if you want to check the transfer learning results out, you may skip
this script and go straight ahead to "3-embed-patches.py".
However, make sure you are passing `pre_weights="imagenet"`.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os
from collections import Counter
from math import ceil

import numpy as np
from PIL import ImageFile
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

ImageFile.LOAD_TRUNCATED_IMAGES = True

ex = Experiment('train-network-multilabel')

ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    data_dir = "/datasets/pbn/random_299/"
    batch_size = 64
    image_shape = [299, 299, 3]
    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = ()
    pooling = 'avg'
    train_shuffle = True
    dataset_train_seed = 12
    valid_shuffle = False
    dataset_valid_seed = 98
    device = "/gpu:0"
    train_info = '/datasets/pbn/train_info.csv'

    classes = None
    num_classes = 1763

    opt_params = {'lr': .001}
    dropout_p = 0.2
    ckpt_file = './ckpt/pbn,%s,multilabel.hdf5' % architecture.lower()
    resuming_from_ckpt_file = ckpt_file
    epochs = 500
    steps_per_epoch = None
    validation_steps = None
    workers = 8
    use_multiprocessing = False
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_file = './logs/2-train-network/pbn,%s,multilabel' % architecture.lower()
    first_trainable_layer = None
    first_reset_layer = None
    class_weight = None


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


@ex.automain
def run(image_shape, data_dir, train_shuffle, dataset_train_seed,
        valid_shuffle, dataset_valid_seed,
        classes, num_classes, train_info,
        architecture, weights, batch_size, last_base_layer, use_gram_matrix,
        pooling, dense_layers,
        device, opt_params, dropout_p, resuming_from_ckpt_file, ckpt_file,
        steps_per_epoch,
        epochs, validation_steps, workers, use_multiprocessing, initial_epoch,
        early_stop_patience,
        tensorboard_file, first_trainable_layer, first_reset_layer,
        class_weight):
    import tensorflow as tf
    from keras import callbacks, optimizers, backend as K
    from keras.preprocessing.image import ImageDataGenerator

    from connoisseur.models import build_model
    from connoisseur.utils import get_preprocess_fn
    from connoisseur.datasets.painter_by_numbers import load_labels

    os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    y, fs, encoders = load_labels(train_info)
    label_map = dict(zip([os.path.splitext(f)[0] for f in fs], y))

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=.2,
        rotation_range=.2,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=get_preprocess_fn(architecture))

    train_data = g.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=image_shape[:2], classes=classes, class_mode='sparse',
        batch_size=batch_size, shuffle=train_shuffle, seed=dataset_train_seed)

    fs = [os.path.basename(f).split('-')[0] for f in train_data.filenames]
    train_data.classes = np.array([label_map[f] for f in fs])

    valid_data = g.flow_from_directory(
        os.path.join(data_dir, 'valid'),
        target_size=image_shape[:2], classes=classes, class_mode='sparse',
        batch_size=batch_size, shuffle=valid_shuffle, seed=dataset_valid_seed)

    fs = [os.path.basename(f).split('-')[0] for f in valid_data.filenames]
    valid_data.classes = np.array([label_map[f] for f in fs])

    del y, fs, encoders

    if class_weight == 'balanced':
        # class_weight = get_class_weights(train_data.classes)
        raise ValueError('class_weight is still a little confusing in '
                         'this multi-label problems')

    if steps_per_epoch is None:
        steps_per_epoch = ceil(train_data.n / batch_size)
    if validation_steps is None:
        validation_steps = ceil(valid_data.n / batch_size)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture,
                            weights=weights, dropout_p=dropout_p,
                            classes=num_class,
                            last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers,
                            predictions_activation='sigmoid')

        layer_names = [l.name for l in model.layers]

        if first_trainable_layer:
            if first_trainable_layer not in layer_names:
                raise ValueError('%s is not a layer in the model: %s' % (
                    first_trainable_layer, layer_names))

            _trainable = False
            for layer in model.layers:
                if layer.name == first_trainable_layer:
                    _trainable = True
                layer.trainable = _trainable
            del _trainable

        model.compile(optimizer=optimizers.Adam(**opt_params),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')

        if resuming_from_ckpt_file:
            print('re-loading weights...')
            model.load_weights(resuming_from_ckpt_file)

        if first_reset_layer:
            if first_reset_layer not in layer_names:
                raise ValueError('%s is not a layer in the model: %s'
                                 % (first_reset_layer, layer_names))
            print('first layer to have its weights reset:', first_reset_layer)
            random_model = build_model(image_shape, architecture=architecture,
                                       weights=None, dropout_p=dropout_p,
                                       classes=num_class,
                                       last_base_layer=last_base_layer,
                                       use_gram_matrix=use_gram_matrix,
                                       dense_layers=dense_layers,
                                       predictions_activation='sigmoid')
            _reset = False
            for layer, random_layer in zip(model.layers, random_model.layers):
                if layer.name == first_reset_layer:
                    _reset = True
                if _reset:
                    layer.set_weights(random_layer.get_weights())
            del random_model

            model.compile(optimizer=optimizers.Adam(**opt_params),
                          metrics=['accuracy'],
                          loss='binary_crossentropy')

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                verbose=1,
                validation_data=valid_data, validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                class_weight=None,
                workers=workers, use_multiprocessing=use_multiprocessing,
                callbacks=[
                    # callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 50) * opt_params['lr']),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(
                        early_stop_patience // 3)),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file,
                                          batch_size=batch_size),
                    callbacks.ModelCheckpoint(ckpt_file, save_best_only=True,
                                              verbose=1),
                ])

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
