"""3 Evaluate SVM.

Evaluate the SVM trained over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sacred import Experiment
from sklearn import metrics
from sklearn.externals import joblib

from connoisseur.datasets import group_by_paintings
from connoisseur.fusion import Fusion, strategies

matplotlib.use('agg')

from connoisseur.datasets import load_pickle_data

tf.logging.set_verbosity(tf.logging.DEBUG)

ex = Experiment('generate-svm-predictions')


@ex.config
def config():
    data_dir = '/datasets/vangogh-test-recaptures/recaptures-vangogh-museum-lower-quality/original/patches/random_299'
    ckpt = '/work/vangogh/ckpt/min-gradient-svm.pkl'
    results_file_name = 'report.json'
    group_patches = True
    group_recaptures = False
    phases = ['test']
    classes = None
    layer = 'global_average_pooling2d_1'
    max_patches = None


def evaluate(model, x, y, names,
             group_patches=False,
             group_recaptures=False,
             max_patches=None):
    try:
        probabilities = model.predict_proba(x)
        labels = np.argmax(probabilities, axis=-1)
        hyperplane_distance = None
        multi_class = True
    except AttributeError:
        probabilities = None
        labels = model.predict(x)
        hyperplane_distance = model.decision_function(x)
        multi_class = len(model.classes_) > 2

    p = probabilities if probabilities is not None else hyperplane_distance

    score = metrics.accuracy_score(y, labels)
    cm = metrics.confusion_matrix(y, labels)
    print('score using raw strategy:', score, '\n',
          metrics.classification_report(y, labels),
          '\nConfusion matrix:\n', cm)

    results = {
        'samples': names.tolist(),
        'labels': y.tolist(),
        'evaluations': [{
            'strategy': 'raw',
            'score': score,
            'p': labels.tolist(),
        }]
    }

    if group_patches:
        labels, p, y, names = group_by_paintings(labels, p, y, names=names, max_patches=max_patches)
        y = np.asarray([_y[0] for _y in y])

        patches = p.shape[1] if len(p.shape) > 1 else 'all'

        for strategy_tag in ('sum', 'mean', 'farthest', 'most_frequent'):
            strategy = getattr(strategies, strategy_tag)

            p_ = (Fusion(strategy=strategy, multi_class=multi_class)
                  .predict(probabilities=p if probabilities is not None else None,
                           hyperplane_distance=p if hyperplane_distance is not None else None,
                           labels=labels))
            score = metrics.accuracy_score(y, p_)
            print('score using', strategy_tag, 'strategy:', score, '\n',
                  metrics.classification_report(y, p_),
                  '\nConfusion matrix:\n',
                  metrics.confusion_matrix(y, p_), '\n',
                  'samples incorrectly classified:', names[p_ != y])

            if group_recaptures:
                print('combined recaptures score:')
                recaptures = np.asarray(['-'.join(n.split('-')[:-1]) for n in names])

                rp = (pd.Series(p_, name='p')
                      .groupby(recaptures)
                      .apply(lambda _x: _x.value_counts().index[0]))
                ry = pd.Series(y, name='y').groupby(recaptures).first()
                ryp = pd.concat([rp, ry], axis=1)
                misses = ryp[ryp['y'] != ryp['p']].index.values

                score = metrics.accuracy_score(ry, rp)
                print('score using', strategy_tag, 'strategy:', score, '\n',
                      metrics.classification_report(ry, rp),
                      '\nConfusion matrix:\n',
                      metrics.confusion_matrix(ry, rp), '\n',
                      'samples incorrectly classified:', misses)

            results['evaluations'].append({
                'strategy': strategy_tag,
                'score': score,
                'p': p_.tolist(),
                'patches': patches
            })

    return results


@ex.automain
def run(_run, data_dir, phases, classes, layer, ckpt,
        results_file_name, group_patches, group_recaptures,
        max_patches):
    report_dir = _run.observers[0].dir

    print('loading model...', end=' ')
    model = joblib.load(ckpt)
    print('done.')

    print('loading data...', end=' ')
    data = load_pickle_data(data_dir=data_dir, phases=phases, chunks=(0, 1), classes=classes, layers=[layer])
    print('done.')

    results = []

    for p in phases:
        print('\n# %s evaluation' % p)
        x, y, names = data[p]
        x = x[layer]
        x = x.reshape(x.shape[0], -1)

        layer_results = evaluate(model, x, y, names,
                                 group_patches=group_patches,
                                 group_recaptures=group_recaptures,
                                 max_patches=max_patches)
        layer_results['phase'] = p
        layer_results['layer'] = layer
        results.append(layer_results)

    with open(os.path.join(report_dir, results_file_name), 'w') as file:
        json.dump(results, file)
