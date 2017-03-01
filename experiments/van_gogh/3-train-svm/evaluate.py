"""3 Evaluate SVM.

Evaluate the SVM trained over the features extracted with InceptionV3 in
`../2-transform-inception/run.py` script.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from connoisseur.fusion import SkLearnFusion

ex = Experiment('3-evaluate-svm')


@ex.config
def config():
    data_dir = '/datasets/ldavid/van_gogh/vgdb_2016'
    nb_patches = 50
    ckpt_file_name = '/work/ldavid/van_gogh/3-train-svm/opt-model.pkl'


def load_data(data_dir, phases=None, share_val_samples=None,
              random_state=None):
    data = {}
    phases = phases or ('train', 'valid', 'test')
    for p in phases:
        try:
            with open(os.path.join(data_dir, '%s.pickle' % p), 'rb') as f:
                d = pickle.load(f)
                data[p] = d['data'], d['target'], d['names']
        except IOError:
            continue

    if 'valid' not in data and share_val_samples:
        # Separate train and valid sets.
        X, y, names = data['train']
        (X_train, X_valid,
         y_train, y_valid,
         names_train, names_valid) = train_test_split(
            X, y, names, test_size=share_val_samples,
            random_state=random_state)

        data['train'] = X_train, y_train, names_train
        data['valid'] = X_valid, y_valid, names_valid

    return data


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


def crop_or_pad_patch_count(X, nb_patches, shuffle=True):
    """Crop or Pad Patches Count.

    Paintings have different sizes and, therefore, are subdivided in a
    different number of patches. This function normalizes the patch count
    for each painting by either duplicating or removing them.
    """
    Z = []
    for x in X:
        if shuffle: np.random.shuffle(x)

        if x.shape[0] < nb_patches:
            _x = x
            while x.shape[0] < nb_patches:
                x = np.concatenate((x, _x))
        x = x[:nb_patches]
        Z.append(x)
    return np.array(Z, copy=False)


def evaluate(model, x, y, names, nb_patches):
    x, y, names = group_by_paintings(x, y, names)
    x = crop_or_pad_patch_count(x, nb_patches=nb_patches)

    for strategy in ('farthest', 'sum', 'most_frequent'):
        f = SkLearnFusion(model, strategy=strategy)
        p = f.predict(x)
        accuracy_score = metrics.accuracy_score(y, p)
        print('score using', strategy, 'strategy: %.2f' % accuracy_score, '\n',
              metrics.classification_report(y, p),
              '\nConfusion matrix:\n',
              metrics.confusion_matrix(y, p))
        print('samples incorrectly classified:', names[p != y])


def evaluate_all(model, data, nb_patches):
    for phase in ('train', 'valid', 'test'):
        x, y, names = data[phase]
        x = x.reshape(x.shape[0], -1)
        y = np.argmax(y, -1)

        print('\n# %s evaluation' % phase)
        evaluate(model, x, y, names, nb_patches=nb_patches)


@ex.automain
def run(data_dir, nb_patches, ckpt_file_name):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    print('loading model...', end=' ')
    model = joblib.load(ckpt_file_name)
    print('done.')

    print('loading data...', end='')
    data = load_data(data_dir=data_dir)
    print('done.')

    evaluate_all(model, data, nb_patches=nb_patches)
