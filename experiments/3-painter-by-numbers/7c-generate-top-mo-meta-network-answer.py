"""Generate Top Network Answer for Painter-by-Numbers.

If the network trained in 4-train-top-network's limbs are too big, the test
activity will take too much time (I estimated 23 days using a Titan X).
In this case, encode the test data using 4-embed-patches.py and the limb
weights. Then use this script to load the trained siamese network, clip
its limbs and  test from the encoded data instead (it will take a few hours).

Encodes each test paintings' pair into probabilities and save it on a file.
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
from keras import backend as K
from keras.utils import Sequence
from sacred import Experiment
from sklearn import metrics

from connoisseur.datasets import load_pickle_data, group_by_paintings
from connoisseur.models import build_siamese_top_meta

tf.logging.set_verbosity(tf.logging.ERROR)
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)

ex = Experiment('generate-top-network-answer-for-painter-by-numbers')


@ex.config
def config():
    data_dir = '/work/datasets/pbn/random_299/'
    submission_info = '/datasets/pbn/submission_info.csv'
    solution = '/datasets/pbn/solution_painter.csv'
    chunks = [0, 1]
    joint_weights = '/work/painter-by-numbers/ckpt/joint.hdf5'
    patches = 50
    batch_size = 10240  # seriously
    dense_layers = ()
    device = "/gpu:0"
    joint = 'multiply'
    ckpt = './ckpt/siamese.hdf5'

    estimator_type = 'probability'
    results_file = 'results.json'
    submission_file = 'answer-{strategy}.csv'

    use_multiprocessing = False
    workers = 1

    outputs_meta = [
        dict(n='artist', u=1584, e=2048, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='style', u=135, e=256, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='genre', u=42, e=128, j='multiply', a='softmax', l='binary_crossentropy', m='accuracy'),
        dict(n='date', u=1, e=32, j='l2', a='linear', l='mse', m='mae')
    ]


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
    def __init__(self, inputs, names, pairs, labels, batch_size):
        self.pairs = pairs
        self.labels = labels
        self.batch_size = batch_size
        self.inputs = inputs

        self.samples, self.patches = next(iter(inputs.values())).shape[:2]
        self.samples_map = dict(zip(names, list(range(self.samples))))

    def __len__(self):
        return int(ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        xb = self.pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        yb = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        zb = {
            '%s_%s' % (o, limb): np.concatenate([_x[self.samples_map[ix]] for ix in e])
            for limb, e in zip('ab', (xb[:, 0], xb[:, 1]))
            for o, _x in self.inputs.items()
        }

        yb = np.repeat(yb, self.patches)
        return zb, yb


@ex.automain
def run(_run, data_dir, patches, estimator_type, submission_info, solution,
        batch_size, dense_layers, device, ckpt,
        results_file, submission_file, use_multiprocessing, workers,
        joint_weights, outputs_meta, chunks):
    report_dir = _run.observers[0].dir

    with tf.device(device):
        print('building...')
        model = build_siamese_top_meta(outputs_meta,
                                       joint_weights=joint_weights,
                                       dense_layers=dense_layers)
        model.summary()
        print('loading weights from', ckpt)
        model.load_weights(ckpt)

        print('loading submission and solution...')
        pairs = pd.read_csv(submission_info, quotechar='"', delimiter=',').values[:, 1:]
        labels = pd.read_csv(solution, quotechar='"', delimiter=',').values[:, 1:].flatten()

        print('loading sequential predictions...')
        d = load_pickle_data(data_dir,
                             phases=['test'], keys=['data', 'names'],
                             chunks=chunks)
        d, names = d['test']

        print('signal names:', d.keys())

        inputs = [d[e['n']] for e in outputs_meta]
        del d
        *inputs, names = group_by_paintings(*inputs, names=names)
        inputs = {o['n']: inputs[ix] for ix, o in enumerate(outputs_meta)}

        names = np.asarray([n.split('/')[1] + '.jpg' for n in names])

        # All outputs should have the same amount of patches.
        assert [i.shape[1] for i in inputs.values()]
        print('test data inputs shape:', [s.shape for s in inputs.values()])

        print('\n# test evaluation')
        test_data = ArrayPairsSequence(inputs, names, pairs, labels, batch_size)
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
