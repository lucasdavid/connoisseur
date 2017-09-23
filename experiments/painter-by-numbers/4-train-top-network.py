"""Train a network on top of the network trained on Painters-by-numbers.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
from collections import Counter
from math import ceil

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment('train-top-network')

ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    data_dir = "/datasets/vangogh/"
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

    classes = None

    opt_params = {'lr': .001}
    dropout_p = 0.2
    ckpt = './ckpt/pbn_inception_dense3_sigmoid.hdf5'
    resuming_from_base_ckpt = ckpt
    epochs = 100
    steps_per_epoch = None
    validation_steps = None
    workers = 8
    use_multiprocessing = False
    initial_epoch = 0
    early_stop_patience = 30
    tensorboard_file = './logs/2-train-top-network/'
    trainable_base_model = False
    class_weight = 'balanced'


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


@ex.automain
def run(image_shape, data_dir, train_shuffle, dataset_train_seed, valid_shuffle, dataset_valid_seed,
        classes,
        architecture, weights, batch_size, last_base_layer, pooling,
        device, opt_params, dropout_p, resuming_from_base_ckpt, ckpt, steps_per_epoch,
        epochs, validation_steps, workers, use_multiprocessing, initial_epoch, early_stop_patience,
        use_gram_matrix, dense_layers,
        tensorboard_file, trainable_base_model, class_weight):
    import os

    import tensorflow as tf
    from PIL import ImageFile
    from keras import Input, layers, callbacks, optimizers, backend as K
    from keras.engine import Model
    from keras.preprocessing.image import ImageDataGenerator

    from connoisseur.models import build_model
    from connoisseur.utils import get_preprocess_fn
    from connoisseur.utils.image import PairsDirectoryIterator

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    preprocess_input = get_preprocess_fn(architecture)

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=.2,
        rotation_range=.2,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=preprocess_input)

    train_data = PairsDirectoryIterator(
        os.path.join(data_dir, 'train'), g,
        target_size=image_shape[:2], classes=classes,
        batch_size=batch_size, shuffle=train_shuffle, seed=dataset_train_seed)

    valid_data = PairsDirectoryIterator(
        os.path.join(data_dir, 'valid'), g,
        target_size=image_shape[:2], classes=classes,
        batch_size=batch_size, shuffle=valid_shuffle, seed=dataset_valid_seed)

    if class_weight == 'balanced':
        class_weight = get_class_weights(train_data.classes)

    if steps_per_epoch is None:
        steps_per_epoch = ceil(train_data.n / batch_size)
    if validation_steps is None:
        validation_steps = ceil(valid_data.n / batch_size)

    with tf.device(device):
        print('building...')
        ia, ib = Input(shape=image_shape), Input(shape=image_shape)
        base_model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                                 classes=train_data.num_class, last_base_layer=last_base_layer,
                                 use_gram_matrix=use_gram_matrix, pooling=pooling,
                                 dense_layers=dense_layers)
        base_model.trainable = trainable_base_model

        ya = base_model(ia)
        yb = base_model(ib)

        x = layers.multiply([ya, yb])
        x = layers.Dense(2018, activation='relu')(x)
        x = layers.Dropout(dropout_p)(x)
        x = layers.Dense(2018, activation='relu')(x)
        x = layers.Dropout(dropout_p)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[ia, ib], outputs=x)

        if resuming_from_base_ckpt:
            print('re-loading weights...')
            model.load_weights(resuming_from_base_ckpt, by_name=True)

        model.compile(optimizer=optimizers.Adam(**opt_params),
                      metrics=['accuracy'],
                      loss='binary_crossentropy')

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1,
                validation_data=valid_data, validation_steps=validation_steps,
                initial_epoch=initial_epoch,
                class_weight=class_weight,
                workers=workers, use_multiprocessing=use_multiprocessing,
                callbacks=[
                    # callbacks.LearningRateScheduler(lambda epoch: .5 ** (epoch // 10) * opt_params['lr']),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_file, batch_size=batch_size),
                    callbacks.ModelCheckpoint(ckpt, save_best_only=True, save_weights_only=True, verbose=1),
                ])

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
