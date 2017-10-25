"""Transfer and/or fine-tune models on a dataset.

Uses an architecture to train over a dataset.
Image patches are loaded directly from the disk.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import matplotlib

from connoisseur.utils.image import MultipleOutputsSequence

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import re
from math import ceil
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
from keras import callbacks, optimizers, backend as K
from keras import layers, Model
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, Imputer, OneHotEncoder

from connoisseur.models import build_model
from connoisseur.utils import get_preprocess_fn

ex = Experiment('train-network-multiple-predictions')

ex.captured_out_filter = apply_backspaces_and_linefeeds
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
    dataset_train_seed = 12
    dataset_valid_seed = 98
    train_shuffle = True
    valid_shuffle = False
    train_info = '/datasets/pbn/train_info.csv'
    classes = None

    architecture = 'InceptionV3'
    tag = 'pbn_%s_multilabel' % architecture.lower()
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    dense_layers = ()
    pooling = 'avg'
    ckpt_file = '%s.hdf5' % tag
    num_classes = 1763

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
    early_stop_patience = 30
    tensorboard_tag = 'train-multilabel-network-%s' % tag
    first_trainable_layer = None
    first_reset_layer = None


@ex.automain
def run(_run, image_shape, data_dir, train_shuffle, dataset_train_seed,
        valid_shuffle, dataset_valid_seed,
        classes, num_classes, train_info,
        architecture, weights, batch_size, last_base_layer, use_gram_matrix,
        pooling, dense_layers,
        device, opt_params, dropout_p, resuming_from, ckpt_file,
        steps_per_epoch,
        epochs, validation_steps, workers, use_multiprocessing, initial_epoch,
        early_stop_patience,
        tensorboard_tag, first_trainable_layer, first_reset_layer):
    try:
        report_dir = _run.observers[0].dir
    except IndexError:
        report_dir = './logs/_unlabeled'

    # reading data.
    print('reading train-info')
    info = pd.read_csv(train_info, quotechar='"', delimiter=',')

    y = {'names': info['filename'].values}
    categorical_outputs = ('artist', 'style', 'genre')

    for p in ('artist', 'style', 'genre', 'date'):
        y[p] = {}
        if p in categorical_outputs:
            p_enc = LabelEncoder()
            y[p]['encoder'] = p_enc
            _y = p_enc.fit_transform(info[p].apply(str))
            flow = make_pipeline(Imputer(strategy='median'),
                                 OneHotEncoder(sparse=False))
        else:
            if p == 'date':
                expression = '.*(\d{4})[\.0]?'
                _y = []
                for e in info[p]:
                    try:
                        _y += [e if isinstance(e, float) else float(re.match(expression, e).group(1))]
                    except AttributeError:
                        _y += [np.nan]
                _y = np.asarray(_y)
            else:
                _y = info[p].apply(float)

            flow = make_pipeline(Imputer(strategy='median'))

        _y = flow.fit_transform(_y.reshape(-1, 1))
        y[p]['values'] = _y
        y[p]['flow'] = flow

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=.2,
        rotation_range=.2,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=get_preprocess_fn(architecture))

    train_data = MultipleOutputsSequence(
        os.path.join(data_dir, 'train'), y, g,
        batch_size=batch_size, target_size=image_shape[:2], classes=classes)

    valid_data = MultipleOutputsSequence(
        os.path.join(data_dir, 'train'), y, g,
        batch_size=batch_size, target_size=image_shape[:2], classes=classes)

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(valid_data)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture,
                            weights=weights, dropout_p=dropout_p,
                            classes=num_classes,
                            last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers,
                            predictions_activation='sigmoid')
        x = model.get_layer(index=-1).output

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

        y = [layers.Dense(1584, activation='softmax', name='painters')(x),
             layers.Dense(112, activation='softmax', name='styles')(x),
             layers.Dense(10, activation='softmax', name='genres')(x),
             layers.Dense(1, activation='linear', name='years')(x)]

        model = Model(inputs=model.input, outputs=y)
        model.compile(optimizer=optimizers.Adam(**opt_params),
                      loss={
                          'painters': 'categorical_crossentropy',
                          'styles': 'categorical_crossentropy',
                          'genres': 'categorical_crossentropy',
                          'years': 'mse'
                      },
                      metrics={'painters': 'accuracy', 'styles': 'accuracy', 'genres': 'accuracy', 'years': 'mse'})

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
                    callbacks.ReduceLROnPlateau(min_lr=1e-10, patience=int(early_stop_patience // 3)),
                    callbacks.EarlyStopping(patience=early_stop_patience),
                    callbacks.TensorBoard(tensorboard_tag, batch_size=batch_size),
                    callbacks.ModelCheckpoint(ckpt_file, save_best_only=True, verbose=1),
                    callbacks.TensorBoard(os.path.join(report_dir, tensorboard_tag), batch_size=batch_size),
                    callbacks.ModelCheckpoint(os.path.join(report_dir, ckpt_file), save_best_only=True, verbose=1),
                ])

        except KeyboardInterrupt:
            print('interrupted by user')
        else:
            print('done')
