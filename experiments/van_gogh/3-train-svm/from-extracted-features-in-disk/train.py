"""3 Train SVM.

Train an SVM over the features extracted with InceptionV3 in
`../2-transform-inception/run.py` script.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import os
import pickle

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

ex = Experiment('4-train-svm')


@ex.config
def config():
    data_dir = '/datasets/ldavid/van_gogh/vgdb_2016'
    nb_samples_used = 10680
    nb_patches = 50
    grid_searching = False
    param_grid = {
        'C': [.1, 1., 10.],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced', None]
    }
    n_jobs = 4
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


@ex.automain
def run(data_dir, nb_samples_used, nb_patches, grid_searching,
        param_grid, n_jobs, ckpt_file_name):
    os.makedirs(os.path.dirname(ckpt_file_name), exist_ok=True)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    data = load_data(data_dir=data_dir, phases=('train', 'valid'))
    # Concatenate train and valid data.
    x, y, _ = data['train']
    X_valid, y_valid, _ = data['valid']
    x, y = map(np.concatenate, ((x, X_valid), (y, y_valid)))
    del data, X_valid, y_valid

    # Training set is too bing. Sub-sample it.
    samples = np.arange(x.shape[0])
    np.random.shuffle(samples)
    samples = samples[:nb_samples_used]
    x = x[samples]
    y = y[samples]

    # Flat the features, which are 3-rank tensors
    # at the end of InceptionV3's convolutions.
    x = x.reshape(x.shape[0], -1)
    # Convert one-hot encoding to labels.
    y = np.argmax(y, -1)

    uniques, counts = np.unique(y, return_counts=True)
    print('occurrences:', dict(zip(uniques, counts)))

    model = Pipeline([
        ('pca', PCA(n_components=.99, random_state=4120)),
        ('svc', SVC(class_weight='balanced', random_state=24))
    ])
    # model = SVC(class_weight='balanced', random_state=24)

    if grid_searching:
        print('grid searching...', end=' ')
        grid = GridSearchCV(SVC(), param_grid=param_grid, n_jobs=n_jobs)
        grid.fit(x, y)
        model = grid.best_estimator_
    else:
        print('training...', end=' ')
        model.fit(x, y)

    print('done.')

    print('saving model...', end=' ')
    joblib.dump(model, ckpt_file_name)
    print('done.')
