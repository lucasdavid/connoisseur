"""Generate Network Answer for Painter-by-Numbers.

This script is separate from usual-ml-pipeline because the test data in
Painter-by-numbers is not a classification task.

This script takes the already encoded test paintings, compute their pair
probabilities and save it on a file. Finally, This file can be given to
6-generate-submission-file.py script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json
import os

import numpy as np
import pandas as pd
from sacred import Experiment
from scipy import stats
from sklearn import metrics

ex = Experiment('generate-meta-network-answer-for-painter-by-numbers')


@ex.config
def config():
    data_dir = '/datasets/pbn/random_299/'
    submission_info_path = '/datasets/pbn/submission_info.csv'
    solution_path = '/datasets/pbn/solution_painter.csv'
    results_file_name = './results-pbn_random_299_inception_auc.json'
    binary_strategy = 'dot'


def evaluate(probabilities, y, names, pairs, binary_strategy):
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
def run(data_dir, submission_info_path, solution_path, binary_strategy, results_file_name):
    from connoisseur.datasets import load_pickle_data

    pairs = pd.read_csv(submission_info_path, quotechar='"', delimiter=',').values[:, 1:]
    y = pd.read_csv(solution_path, quotechar='"', delimiter=',').values[:, 1:]

    pairs = [['unknown/' + os.path.splitext(a)[0], 'unknown/' + os.path.splitext(b)[0]]
             for a, b in pairs]

    results = []
    for phase in ['test']:
        print('\n# %s evaluation' % phase)

        d = load_pickle_data(data_dir, phases=['test'], keys=['data', 'names'])
        d, names = d['test']
        probabilities = d['predictions']
        del d

        layer_results = evaluate(probabilities, y, names, pairs, binary_strategy)
        layer_results['phase'] = phase
        results.append(layer_results)

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
