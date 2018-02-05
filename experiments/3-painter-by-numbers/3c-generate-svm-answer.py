"""Generate SVM Answer for Painter-by-Numbers.

This script takes an SVM trained over a network, encodes each test paintings'
pair into probabilities and save it on a file. Finally, This file can be
given to 6-generate-submission-file.py script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json
import os

import matplotlib

matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sacred import Experiment
from sklearn import metrics
from sklearn.externals import joblib

from connoisseur.fusion import strategies
from connoisseur.datasets import group_by_paintings
from connoisseur.fusion.base import Fusion

ex = Experiment('generate-svm-answer-for-painter-by-numbers')


@ex.config
def config():
    ex_tag = 'random_299_inception_pca:0.99_svm'
    data_dir = '/datasets/vangogh/'
    phases = ['train', 'valid', 'test']
    classes = None
    layer = 'avg_pool'
    ckpt_file_name = ex_tag + '.pkl'
    results_file_name = './results-%s.json' % ex_tag
    submission_info_path = '/datasets/pbn/submission_info.csv'
    solution_path = '/datasets/pbn/solution_painter.csv'


def plot_confusion_matrix(cm, labels, name='cm.jpg', **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, **kwargs)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(name)


def evaluate(model, x, y, names, pairs, tag):
    results = {'samples': pairs, 'labels': y.tolist(), 'evaluations': []}
    x, _, names = group_by_paintings(x, None, names)
    samples, patches, features = x.shape
    name_indices = {n: i for i, n in enumerate(names)}
    pair_indices = np.array([[name_indices[a], name_indices[b]] for a, b in pairs]).T

    try:
        probabilities = model.predict_proba(x.reshape(-1, features)).reshape(samples, patches, -1)
        labels = None
        hyperplane_distance = None
        multi_class = True
    except AttributeError:
        probabilities = None
        labels = model.predict(x.reshape(-1, features)).reshape(samples, patches)
        hyperplane_distance = model.decision_function(x.reshape(-1, features)).reshape(samples, patches, -1)
        multi_class = len(model.classes_) > 2
        if not multi_class:
            hyperplane_distance = np.squeeze(hyperplane_distance, axis=-1)

    for strategy_tag in ('sum', 'mean', 'farthest', 'most_frequent'):
        strategy = getattr(strategies, strategy_tag)

        p = Fusion(strategy=strategy, multi_class=multi_class).predict(probabilities=probabilities,
                                                                       hyperplane_distance=hyperplane_distance,
                                                                       labels=labels)
        p = p[pair_indices]
        p = (p[0] == p[1]).astype(np.float)
        score = metrics.accuracy_score(y, p)
        print('score using', strategy_tag, 'strategy:', score, '\n',
              metrics.classification_report(y, p),
              '\nConfusion matrix:\n',
              metrics.confusion_matrix(y, p))
        print('samples incorrectly classified:', names[p != y], '\n')

        results['evaluations'].append({
            'strategy': strategy_tag,
            'score': score,
            'p': p.tolist()
        })

    return results


@ex.automain
def run(ex_tag, data_dir, phases, classes, layer, ckpt_file_name,
        results_file_name, submission_info_path, solution_path):
    from connoisseur.datasets import load_pickle_data

    print('loading model...', end=' ')
    model = joblib.load(ckpt_file_name)
    print('done.')

    print('loading data...', end=' ')
    data = load_pickle_data(data_dir=data_dir, phases=phases, chunks=(0, 1), classes=classes, layers=[layer])
    print('done.')

    results = []

    pairs = pd.read_csv(submission_info_path, quotechar='"', delimiter=',').values[:, 1:]
    y = pd.read_csv(solution_path, quotechar='"', delimiter=',').values[:, 1:]

    pairs = [['unknown/' + os.path.splitext(a)[0], 'unknown/' + os.path.splitext(b)[0]]
             for a, b in pairs]

    for p in phases:
        print('\n# %s evaluation' % p)
        x, _, names = data[p]
        x = x[layer]
        x = x.reshape(x.shape[0], -1)

        layer_results = evaluate(model, x, y, names,
                                 pairs=pairs,
                                 tag=ex_tag)
        layer_results['phase'] = p
        layer_results['layer'] = layer
        results.append(layer_results)

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
