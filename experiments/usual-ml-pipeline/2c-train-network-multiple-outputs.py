"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
from keras import layers, Model, callbacks, optimizers, backend as K
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment, utils as sacred_utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder

from connoisseur.models import build_model
from connoisseur.utils import get_preprocess_fn
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
    validation_steps = None
    workers = 8
    use_multiprocessing = False
    initial_epoch = 0
    early_stop_patience = 20
    tensorboard_tag = 'train-multiple-outputs-network-%s' % tag
    first_trainable_layer = None


def to_year(e):
    year_pattern = r'(?:\w*[\s\.])?(?P<year>\d{3,4})(?:\.0?)?$'
    try:
        return e if isinstance(e, float) else re.match(year_pattern, e).group(1)
    except AttributeError:
        print('unknown year', e)
        return np.nan


@ex.automain
def run(_run, data_dir, subdirectories, train_shuffle, valid_shuffle, image_shape, train_info, batch_size,
        architecture, weights, last_base_layer, use_gram_matrix, pooling, dropout_p, device,
        epochs, steps_per_epoch, validation_steps, initial_epoch, opt_params, resuming_from, ckpt_file,
        workers, use_multiprocessing, early_stop_patience, first_trainable_layer, tensorboard_tag):
    try:
        report_dir = _run.observers[0].dir
    except IndexError:
        report_dir = './logs/_unlabeled'

    print('reading train-info...')
    y_train = pd.read_csv(train_info, quotechar='"', delimiter=',')

    categorical_output_names = ['artist', 'style', 'genre']
    output_names = ['artist', 'style', 'genre'] + ['date']
    outputs = {}

    for f in output_names:
        if f in categorical_output_names:
            en = LabelEncoder()
            is_nan = pd.isnull(y_train[f])
            encoded = en.fit_transform(y_train[f].apply(str).str.lower()).astype('float')
            encoded[is_nan] = np.nan

            flow = make_pipeline(Imputer(strategy='most_frequent'),
                                 OneHotEncoder(sparse=False))

        else:
            encoded = y_train[f] if f != 'date' else y_train['date'].apply(to_year)
            encoded = encoded.values
            flow = make_pipeline(Imputer(strategy='mean'))

        outputs[f] = flow.fit_transform(encoded.reshape(-1, 1))

    name_map = {os.path.splitext(n)[0]: i for i, n in enumerate(y_train['filename'])}

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
    units = {o: outputs[o].shape[1] for o in output_names}
    del y_train, categorical_output_names, output_names, outputs, name_map, g

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(valid_data)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture,
                            weights=weights, dropout_p=dropout_p,
                            last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            include_top=False)

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

        x = model.output
        y = [layers.Dense(units['artist'], activation='softmax', name='artist')(x),
             layers.Dense(units['style'], activation='softmax', name='style')(x),
             layers.Dense(units['genre'], activation='softmax', name='genre')(x),
             layers.Dense(units['date'], activation='relu', name='date')(x)]

        model = Model(inputs=model.input, outputs=y)
        model.compile(optimizer=optimizers.Adam(**opt_params),
                      loss={
                          'artist': 'categorical_crossentropy',
                          'style': 'categorical_crossentropy',
                          'genre': 'categorical_crossentropy',
                          'date': 'mse'
                      },
                      metrics={'artist': 'accuracy',
                               'style': 'accuracy',
                               'genre': 'accuracy',
                               'year': 'mse'},
                      loss_weights={'artist': 1.,
                                    'style': 0.8,
                                    'genre': 0.6,
                                    'year': 0.4})
        model.summary()

        if resuming_from:
            print('re-loading weights...')
            model.load_weights(resuming_from)

        print('training from epoch %i...' % initial_epoch)
        try:
            model.fit_generator(
                train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                validation_data=valid_data, validation_steps=validation_steps,
                initial_epoch=initial_epoch, verbose=1,
                workers=workers, use_multiprocessing=use_multiprocessing,
                callbacks=[
                    callbacks.TerminateOnNaN(),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                    callbacks.ModelCheckpoint(os.path.join(report_dir, ckpt_file), save_best_only=True, verbose=1),
                ])

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
