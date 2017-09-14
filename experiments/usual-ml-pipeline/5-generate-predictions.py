"""3 Evaluate SVM.

Evaluate the SVM trained over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn import metrics
from sklearn.externals import joblib

from connoisseur.fusion import SkLearnFusion, strategies

ex = Experiment('5-generate-predictions')


@ex.config
def config():
    ex_tag = 'random_299_inception_pca:0.99_svm'
    data_dir = '/datasets/vangogh/'
    ckpt_file_name = ex_tag + '.pkl'
    results_file_name = './results-%s.json' % ex_tag
    group_patches = True
    nb_patches = 50
    phases = ['train', 'valid', 'test']
    classes = None
    layer = 'avg_pool'


def plot_confusion_matrix(cm, labels, cmap=plt.cm.YlOrRd, name='cm.jpg'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=cmap)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig(name)


def group_by_paintings(x, y, names):
    # Aggregate test patches by their respective paintings.
    _x, _y, _names = [], [], []
    # Remove patches indices, leaving just the painting name.
    clipped_names = np.array(['-'.join(n.split('-')[:-1]) for n in names])
    for name in set(clipped_names):
        s = clipped_names == name
        _x.append(x[s])
        _y.append(y[s][0])
        _names.append(clipped_names[s][0])

    return (np.array(_x, copy=False),
            np.array(_y, copy=False),
            np.array(_names, copy=False))


def evaluate(model, x, y, names, tag, group_patches, nb_patches, phase):
    p = model.predict(x)
    score = metrics.accuracy_score(y, p)
    cm = metrics.confusion_matrix(y, p)
    print('score using raw strategy:', score, '\n',
          metrics.classification_report(y, p),
          '\nConfusion matrix:\n', cm)

    plot_confusion_matrix(cm, [str(i) for i in np.unique(y)],
                          name='-'.join((tag, phase, 'cm.jpg')))

    if group_patches:
        x, y, names = group_by_paintings(x, y, names)

    results = {
        'samples': names.tolist(),
        'labels': y.tolist(),
        'patches_count': nb_patches,
        'evaluations': [{
            'strategy': 'raw',
            'score': score,
            'p': p.tolist(),
        }]
    }

    if group_patches:
        for strategy_tag in ('sum', 'mean', 'farthest', 'most_frequent'):
            strategy = getattr(strategies, strategy_tag)

            p = SkLearnFusion(model, strategy=strategy).predict(x)
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
def run(ex_tag, data_dir, phases, classes, layer, nb_patches, ckpt_file_name,
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
                                 nb_patches=nb_patches,
                                 group_patches=group_patches,
                                 phase=p)
        layer_results['phase'] = p
        layer_results['layer'] = layer
        results.append(layer_results)

    with open(results_file_name, 'w') as file:
        json.dump(results, file)
