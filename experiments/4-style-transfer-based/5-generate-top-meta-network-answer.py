"""Generate Top Network Answer for Painter-by-Numbers.

If the network trained in 4-train-top-network's limbs are too big, the test
activity will take too much time (I estimated 23 days using a Titan X).
In this case, encode the test data using 4-embed-patches.py and the limb
weights. Then use this script to load the trained siamese network, clip
its limbs and  test from the encoded data instead (it will take a few hours).

Encodes each test paintings' pair into probabilities and save it on a file.
Finally, This file can be given to 1-generate-submission-file.py script.

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
from keras import Input, backend as K
from keras.engine import Model
from keras.utils import Sequence
from sacred import Experiment
from sklearn import metrics

from connoisseur.datasets import load_pickle_data, group_by_paintings
from connoisseur.models import build_siamese_model, build_siamese_gram_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)

ex = Experiment('generate-top-network-answer-for-painter-by-numbers')


@ex.config
def config():
    device = "/gpu:0"

    data_dir = "/datasets/pbn/preprocessed/gr/"
    input_shape = [200]
    patches = 50
    estimator_type = 'probability'
    submission_info = '/datasets/pbn/submission_info.csv'
    solution = '/datasets/pbn/solution_painter.csv'
    results_file = 'top-mo.json'
    submission_file = 'answer-{strategy}.csv'
    chunks = [0]

    batch_size = 128
    embedding_units = 1024
    joints = 'multiply'
    include_sigmoid_unit = True

    ckpt = 'top-network.hdf5'
    use_multiprocessing = False
    workers = 1


def evaluate(labels, probabilities, estimator_type):
    print('aggregating patches')
    results = {
        'evaluations': [],
    }

    print('all done, proceeding to fusion')
    probabilities = probabilities.mean(axis=-1)

    if estimator_type == 'distance':
        # Closer distance means more likely the same (1.0)
        probabilities = 1 - probabilities

    probabilities = np.clip(probabilities, 0, 1)
    p = (probabilities > .5).astype(np.float)

    roc_auc = metrics.roc_auc_score(labels, probabilities)
    print('roc auc: ', roc_auc, '\n',
          'accuracy: ', metrics.accuracy_score(labels, p, normalize=True), '\n',
          metrics.classification_report(labels, p), '\n',
          'confusion matrix:\n', metrics.confusion_matrix(labels, p),
          sep='')

    results['evaluations'].append({
        'strategy': 'mean',
        'score': roc_auc,
        'binary_probabilities': probabilities.tolist(),
        'p': p.tolist()
    })

    return results


class ArrayPairsSequence(Sequence):
    def __init__(self, samples, names, pairs, labels, batch_size):
        self.pairs = pairs
        self.labels = labels
        self.batch_size = batch_size
        self.samples_map = dict(zip(names, samples))
        self.samples, self.patches, self.features = samples.shape

    def __len__(self):
        return int(ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        xb = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        yb = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        zb = np.concatenate([[self.samples_map[a], self.samples_map[b]] for a, b in xb], axis=1)
        yb = np.repeat(yb, self.patches)
        return list(zb), yb


@ex.automain
def run(_run, device,
        data_dir, input_shape, patches, estimator_type, submission_info, solution, chunks,
        batch_size,
        embedding_units,
        joints, include_sigmoid_unit, ckpt,
        results_file, submission_file, use_multiprocessing, workers):
    report_dir = _run.observers[0].dir

    with tf.device(device):
        print('building...')

        x = Input(shape=input_shape)
        identity_model = Model(inputs=x, outputs=x)
        model = build_siamese_gram_model(input_shape, architecture=None, dropout_rate=0,
                                         embedding_units=embedding_units,
                                         joints=joints, include_sigmoid_unit=include_sigmoid_unit,
                                         limb=identity_model)
        model.load_weights(ckpt, by_name=True)
        model.summary()

        print('loading submission and solution...')
        pairs = pd.read_csv(submission_info, quotechar='"', delimiter=',').values[:, 1:]
        labels = pd.read_csv(solution, quotechar='"', delimiter=',').values[:, 1:].flatten()

        print('loading sequential predictions...')
        d = load_pickle_data(data_dir, phases=['test'], keys=['data', 'names'], chunks=chunks)
        d, names = d['test']
        samples = d['predictions']
        del d
        samples, names = group_by_paintings(samples, names=names)
        names = np.asarray([n.split('/')[1] + '.jpg' for n in names])

        print('test data shape:', samples.shape)

        print('\n# test evaluation')
        test_data = ArrayPairsSequence(samples, names, pairs, labels, batch_size)
        probabilities = model.predict_generator(
            test_data,
            use_multiprocessing=use_multiprocessing,
            workers=workers, verbose=1).reshape(-1, patches)
        del model
        K.clear_session()

    layer_results = evaluate(labels, probabilities, estimator_type)
    layer_results['phase'] = 'test'
    evaluation_results = [layer_results]

    # generate results file.
    with open(os.path.join(report_dir, results_file), 'w') as file:
        json.dump(evaluation_results, file)

    # generate submission file to Kaggle.
    for v in layer_results['evaluations']:
        predictions_field = 'binary_probabilities' if 'binary_probabilities' in v else 'p'
        p = v[predictions_field]

        with open(submission_file.format(strategy=v['strategy']), 'w') as f:
            f.write('index,sameArtist\n')
            f.writelines(['%i,%f\n' % (i, _p) for i, _p in enumerate(p)])
