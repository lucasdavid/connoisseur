"""Generate Network Answer for Painter-by-Numbers.

This script is separate from usual-ml-pipeline because the test data in
Painter-by-numbers is not a classification task.

This script takes a one-leg network, encodes each test paintings'
pair into probabilities and save it on a file. Finally, This file can be
given to 6-generate-submission-file.py script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import numpy as np
from sacred import Experiment

ex = Experiment('generate-network-answer-for-painter-by-numbers')


@ex.config
def config():
    image_shape = [299, 299, 3]
    data_dir = '/datasets/pbn/random_299/'
    submission_info_path = '/datasets/pbn/submission_info.csv'
    solution_path = '/datasets/pbn/solution_painter.csv'
    data_seed = 12
    classes = None
    n_classes = 1584
    architecture = 'InceptionV3'
    weights = 'imagenet'
    batch_size = 64
    last_base_layer = None
    use_gram_matrix = False
    pooling = 'avg'
    dense_layers = ()
    device = "/gpu:0"
    dropout_p = 0.2
    ckpt_file = './ckpt/pbn_random_299_inception.hdf5'
    results_file_name = './results-pbn_random_299_inception_auc.json'
    binary_strategy = 'dot'


def evaluate(probabilities, y, names, pairs, binary_strategy):
    from scipy import stats
    from sklearn import metrics
    from connoisseur.datasets import group_by_paintings

    print('aggregating patches')
    probabilities, names = group_by_paintings(probabilities, names=names)

    results = {'pairs': pairs,
               'labels': y.tolist(),
               'evaluations': [],
               'names': names.tolist()}

    print('all done, proceeding to fusion')
    probabilities = probabilities.mean(axis=-2)

    print('generating name map')
    name_indices = {n: i for i, n in enumerate(names)}

    if binary_strategy == 'dot':
        binary_strategy = np.dot
    elif binary_strategy == 'pearsonr':
        binary_strategy = lambda _x, _y: stats.pearsonr(_x, _y)[0]
    else:
        raise ValueError('unknown binary strategy %s' % binary_strategy)

    binary_probabilities = np.clip([
        binary_strategy(probabilities[name_indices[a]], probabilities[name_indices[b]])
        for a, b in pairs
    ], 0, 1)

    p = (binary_probabilities > .5).astype(np.float)

    score = metrics.roc_auc_score(y, binary_probabilities)
    print('roc auc score using mean strategy:', score, '\n',
          metrics.classification_report(y, p),
          '\nConfusion matrix:\n',
          metrics.confusion_matrix(y, p))
    print('samples incorrectly classified:', names[p != y], '\n')

    results['evaluations'].append({
        'strategy': 'mean',
        'score': score,
        'probabilities': probabilities.tolist(),
        'binary_probabilities': binary_probabilities.tolist()
    })

    return results


@ex.automain
def run(image_shape, data_dir, submission_info_path, solution_path, data_seed, classes, architecture,
        weights, batch_size, last_base_layer, use_gram_matrix, pooling, dense_layers, device, n_classes,
        dropout_p, ckpt_file, binary_strategy, results_file_name):
    import json
    import os
    from math import ceil

    import pandas as pd
    import tensorflow as tf
    from PIL import ImageFile

    from keras import backend as K
    from keras.preprocessing.image import ImageDataGenerator
    from connoisseur.models import build_model
    from connoisseur.utils import get_preprocess_fn

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    s = tf.Session(config=tf_config)
    K.set_session(s)

    pairs = pd.read_csv(submission_info_path, quotechar='"', delimiter=',').values[:, 1:]
    y = pd.read_csv(solution_path, quotechar='"', delimiter=',').values[:, 1:]

    pairs = [['unknown/' + os.path.splitext(a)[0], 'unknown/' + os.path.splitext(b)[0]]
             for a, b in pairs]

    preprocess_input = get_preprocess_fn(architecture)
    g = ImageDataGenerator(preprocessing_function=preprocess_input)

    with tf.device(device):
        print('building...')
        model = build_model(image_shape, architecture=architecture, weights=weights, dropout_p=dropout_p,
                            classes=n_classes, last_base_layer=last_base_layer,
                            use_gram_matrix=use_gram_matrix, pooling=pooling,
                            dense_layers=dense_layers)

        if ckpt_file:
            print('re-loading weights...')
            model.load_weights(ckpt_file)

        results = []
        for phase in ['test']:
            print('\n# %s evaluation' % phase)

            data = g.flow_from_directory(
                os.path.join(data_dir, phase),
                target_size=image_shape[:2], classes=classes,
                batch_size=batch_size, seed=data_seed,
                shuffle=False)

            steps = ceil(data.n / batch_size)

            probabilities = model.predict_generator(data, steps=steps)
            del model
            K.clear_session()

            layer_results = evaluate(probabilities, y, data.filenames, pairs, binary_strategy)
            layer_results['phase'] = phase
            results.append(layer_results)

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
