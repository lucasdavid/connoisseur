"""3 Evaluate SVM.

Evaluate the SVM trained over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn import metrics
from sklearn.externals import joblib

from connoisseur.fusion import SkLearnFusion

from base import load_data

ex = Experiment('van-gogh/3-train-svm/evaluate')


@ex.config
def config():
    data_dir = '/datasets/van_gogh/vgdb_2016'
    ckpt_file_name = '/work/van_gogh/3-train-svm/opt-svm.pkl'
    nb_patches = 50


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


def evaluate(model, x, y, names, nb_patches):
    x, y, names = group_by_paintings(x, y, names)

    p = model.predict(x.reshape((-1,) + x.shape[2:]))
    score = metrics.accuracy_score(y.repeat(nb_patches), p)
    print('patches accuracy score:', score)

    for strategy in ('sum', 'mean', 'farthest', 'most_frequent'):
        p = SkLearnFusion(model, strategy=strategy).predict(x)
        score = metrics.accuracy_score(y, p)
        print('score using', strategy, 'strategy:', score, '\n',
              metrics.classification_report(y, p),
              '\nConfusion matrix:\n',
              metrics.confusion_matrix(y, p))
        print('samples incorrectly classified:', names[p != y])


@ex.automain
def run(data_dir, nb_patches, ckpt_file_name):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    print('loading model...', end=' ')
    model = joblib.load(ckpt_file_name)
    print('done.')

    print('loading data...', end=' ')
    data = load_data(data_dir=data_dir)
    print('done.')

    for phase in ('train', 'valid', 'test'):
        x, y, names = data[phase]
        x = x.reshape(x.shape[0], -1)
        y = np.argmax(y, -1)

        print('\n# %s evaluation' % phase)
        evaluate(model, x, y, names, nb_patches=nb_patches)
