"""Generate Top Network Answer for Painter-by-Numbers.

This script takes the top network trained by 4-train-top-network.py script,
encodes each test paintings' pair into probabilities and save it on a file.
Finally, This file can be given to 6-generate-submission-file.py script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json
import os
from math import ceil

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import ImageFile
from keras import layers, Input, backend as K
from keras.engine import Model
from keras.preprocessing.image import img_to_array, load_img
from sacred import Experiment
from sklearn import metrics

from connoisseur.models import build_model
from connoisseur.utils import get_preprocess_fn

ImageFile.LOAD_TRUNCATED_IMAGES = True

ex = Experiment('generate-top-network-answer-for-painter-by-numbers')


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = '/datasets/pbn/random_299/'
    submission_info_path = '/datasets/pbn/submission_info.csv'
    solution_path = '/datasets/pbn/solution_painter.csv'
    n_classes = 1584
    architecture = 'InceptionV3'
    weights = 'imagenet'
    patches = 1
    batch_size = 256
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = ()
    device = "/gpu:0"
    dropout_p = 0.2
    ckpt_file = '/work/painter-by-numbers/ckpt/pbn_inception_dense3_sigmoid.hdf5'
    results_file_name = './results-pbn_random_299_inception_dense3_sigmoid.hdf5_auc.json'


def evaluate(probabilities, y, pairs):
    print('aggregating patches')
    results = {
        'pairs': pairs,
        'labels': y.tolist(),
        'evaluations': [],
    }

    print('all done, proceeding to fusion')
    probabilities = probabilities.mean(axis=-1)
    p = (probabilities > .5).astype(np.float)

    score = metrics.roc_auc_score(y, p)
    print('roc auc score using mean strategy:', score, '\n',
          metrics.classification_report(y, p),
          '\nConfusion matrix:\n',
          metrics.confusion_matrix(y, p))

    results['evaluations'].append({
        'strategy': 'mean',
        'score': score,
        'probabilities': probabilities.tolist(),
        'p': p.tolist()
    })

    return results


def pairs_generator(pairs, y, target_size, patches, batch_size, preprocess_input_fn):
    names = []
    for a, b in pairs:
        names += [[a + '-' + i + '.jpg', b + '-' + i + '.jpg'] for i in map(str, range(patches))]

    seen = 0
    while seen < len(names):
        names_batch = names[seen:seen + batch_size]

        x = [[], []]

        for a, b in names_batch:
            x[0].append(img_to_array(load_img(a, target_size=target_size)))
            x[1].append(img_to_array(load_img(b, target_size=target_size)))

        x = list(map(preprocess_input_fn, map(np.array, (x[0], x[1]))))
        yield x, y[seen:seen + batch_size]
        seen += len(names_batch)

    raise StopIteration


@ex.automain
def run(image_shape, data_dir, patches,
        submission_info_path, solution_path,
        architecture, weights, batch_size, last_base_layer, use_gram_matrix, pooling,
        dense_layers, device, n_classes, dropout_p, ckpt_file, results_file_name):
    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    with tf.device(device):
        print('building...')
        ia, ib = Input(shape=image_shape), Input(shape=image_shape)
        base_model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                                 classes=n_classes, last_base_layer=last_base_layer,
                                 use_gram_matrix=use_gram_matrix, pooling=pooling,
                                 dense_layers=dense_layers)
        ya = base_model(ia)
        yb = base_model(ib)
        x = layers.multiply([ya, yb])
        x = layers.Dense(2018, activation='relu')(x)
        x = layers.Dropout(dropout_p)(x)
        x = layers.Dense(2018, activation='relu')(x)
        x = layers.Dropout(dropout_p)(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[ia, ib], outputs=x)

        if ckpt_file:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        pairs = pd.read_csv(submission_info_path, quotechar='"', delimiter=',').values[:, 1:]
        y = pd.read_csv(solution_path, quotechar='"', delimiter=',').values[:, 1:]

        pairs = [[data_dir + 'test/unknown/' + os.path.splitext(a)[0],
                  data_dir + 'test/unknown/' + os.path.splitext(b)[0]]
                 for a, b in pairs]

        print('\n# test evaluation')
        data = pairs_generator(pairs, y, target_size=image_shape, patches=patches,
                               batch_size=batch_size, preprocess_input_fn=get_preprocess_fn(architecture))

        steps = ceil(len(pairs) / batch_size)
        probabilities = model.predict_generator(data, steps=steps).reshape(-1, patches)
        del model
        K.clear_session()

    layer_results = evaluate(probabilities=probabilities, y=y, pairs=pairs)
    layer_results['phase'] = 'test'
    results = [layer_results]

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
