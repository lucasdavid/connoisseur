"""Evaluate Network Multiple Outputs.


Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sacred import Experiment, utils as sacred_utils
from sklearn import metrics

from connoisseur import get_preprocess_fn
from connoisseur.datasets.painter_by_numbers import load_multiple_outputs
from connoisseur.models import build_model
from connoisseur.utils.image import MultipleOutputsDirectorySequence

ex = Experiment('evaluate-network-mo')

ex.captured_out_filter = sacred_utils.apply_backspaces_and_linefeeds
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)


@ex.config
def config():
    data_dir = "/datasets/pbn/patches/random299/"
    batch_size = 64
    image_shape = [299, 299, 3]
    train_shuffle = True
    valid_shuffle = True
    train_info = '/datasets/pbn/train_info.csv'
    subdirectories = None

    architecture = 'InceptionV3'
    weights = 'imagenet'
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'

    device = "/gpu:0"

    resuming_from = None
    workers = 8
    use_multiprocessing = True
    outputs_meta = [
        {'n': 'artist', 'u': 1584, 'a': 'softmax', 'l': 'categorical_crossentropy',
         'm': ['categorical_accuracy', 'top_k_categorical_accuracy'], 'w': .5},
        {'n': 'style', 'u': 135, 'a': 'softmax', 'l': 'categorical_crossentropy',
         'm': ['categorical_accuracy', 'top_k_categorical_accuracy'], 'w': .2},
        {'n': 'genre', 'u': 42, 'a': 'softmax', 'l': 'categorical_crossentropy',
         'm': ['categorical_accuracy', 'top_k_categorical_accuracy'], 'w': .2},
        {'n': 'date', 'u': 1, 'a': 'linear', 'l': 'mse', 'm': 'mae', 'w': .1}
    ]


@ex.automain
def run(_run, data_dir, subdirectories, image_shape, train_info, batch_size,
        architecture, weights, last_base_layer, use_gram_matrix, pooling, device,
        resuming_from, workers, use_multiprocessing, outputs_meta):
    print('reading train-info...')
    outputs, name_map = load_multiple_outputs(train_info, outputs_meta, encode='sparse')

    g = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=.2,
        rotation_range=90,
        height_shift_range=.2,
        width_shift_range=.2,
        fill_mode='reflect',
        preprocessing_function=get_preprocess_fn(architecture))

    train_data = MultipleOutputsDirectorySequence(os.path.join(data_dir, 'train'), outputs, name_map, g,
                                                  batch_size=batch_size, target_size=image_shape[:2],
                                                  subdirectories=subdirectories,
                                                  shuffle=False)
    valid_data = MultipleOutputsDirectorySequence(os.path.join(data_dir, 'valid'), outputs, name_map, g,
                                                  batch_size=batch_size, target_size=image_shape[:2],
                                                  subdirectories=subdirectories,
                                                  shuffle=False)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape,
                            architecture=architecture,
                            weights=weights,
                            last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            include_top=True,
                            classes=[o['u'] for o in outputs_meta],
                            predictions_name=[o['n'] for o in outputs_meta],
                            predictions_activation=[o['a'] for o in outputs_meta])

        if resuming_from:
            print('re-loading weights...')
            model.load_weights(resuming_from)

        for d in (train_data, valid_data):
            p = model.predict_generator(
                d,
                workers=workers,
                use_multiprocessing=use_multiprocessing)

            predictions = [(np.argmax(_p, axis=-1) if o['n'] != 'date' else _p)
                           for _p, o in zip(p, outputs_meta)]

            labels = [outputs[m['n']][d.classes] for m in outputs_meta]

            for n, y, p in zip(outputs, labels, predictions):
                print('Evaluating', n)
                meta = next(o for o in outputs_meta if o['n'] == n)

                if n != 'date':
                    print('accuracy:', metrics.accuracy_score(y, p))
                    print('recall-macro:', metrics.recall_score(y, p, average='macro'))
                    print('f1-macro:', metrics.f1_score(y, p, average='macro'))

                    print(metrics.classification_report(y, p))
                else:
                    _s = meta['f'].named_steps['standardscaler']
                    print('standard_scaler:', _s)
                    print('mae on original space:', metrics.mean_absolute_error(y, p))
                print()
