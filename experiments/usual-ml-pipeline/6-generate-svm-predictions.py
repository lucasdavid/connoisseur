"""3 Evaluate SVM.

Evaluate the SVM trained over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn import metrics
from sklearn.externals import joblib

from connoisseur.datasets import group_by_paintings
from connoisseur.fusion import Fusion, strategies
from connoisseur.utils import plot_confusion_matrix

ex = Experiment('generate-svm-predictions')


@ex.config
def config():
    ex_tag = 'min_299_inception_pca:0.99_svm'
    data_dir = '/datasets/vangogh_preprocessed/random_299/inception/'
    ckpt_file_name = '/work/vangogh/models/min-gradient-svm.pkl'
    results_file_name = './results-%s.json' % ex_tag
    group_patches = True
    phases = ['train', 'test']
    classes = None
    layer = 'avg_pool'


def evaluate(model, x, y, names, tag, group_patches, phase):
    labels = model.predict(x)
    score = metrics.accuracy_score(y, labels)
    cm = metrics.confusion_matrix(y, labels)
    print('score using raw strategy:', score, '\n',
          metrics.classification_report(y, labels),
          '\nConfusion matrix:\n', cm)

    plot_confusion_matrix(cm, [str(i) for i in np.unique(y)],
                          name='-'.join((tag, phase, 'cm.jpg')),
                          cmap='BuPu')

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
        x, y, names = group_by_paintings(x, y, names)

        samples, patches, features = x.shape

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

            p = (Fusion(strategy=strategy, multi_class=multi_class)
                 .predict(probabilities=probabilities, labels=labels,
                          hyperplane_distance=hyperplane_distance))
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
        results_file_name, group_patches):
    from connoisseur.datasets import load_pickle_data
    tf.logging.set_verbosity(tf.logging.DEBUG)

    print('loading model...', end=' ')
    model = joblib.load(ckpt_file_name)
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
                                 tag=ex_tag,
                                 group_patches=group_patches,
                                 phase=p)
        layer_results['phase'] = p
        layer_results['layer'] = layer
        results.append(layer_results)

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
