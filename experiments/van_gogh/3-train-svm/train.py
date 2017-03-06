"""3 Train SVM.

Evaluate the SVM trained over the .
Train an SVM over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from base import load_data

ex = Experiment('van-gogh/3-train-svm/train')


@ex.config
def config():
    data_dir = '/datasets/van_gogh/vgdb_2016'
    ckpt_file_name = '/work/van_gogh/3-train-svm/opt-svm.pkl'
    nb_samples_used = None
    grid_searching = True
    param_grid = {
        'svc__C': [.1, 1., 10.],
        'svc__kernel': ['rbf', 'linear'],
        'svc__class_weight': ['balanced', None]
    }
    cv = None
    n_jobs = 6


@ex.automain
def run(data_dir, nb_samples_used, grid_searching,
        param_grid, cv, n_jobs, ckpt_file_name):
    os.makedirs(os.path.dirname(ckpt_file_name), exist_ok=True)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    data = load_data(data_dir=data_dir, phases=('train', 'valid'))
    # Concatenate train and valid data.
    X, y, _ = data['train']
    X_valid, y_valid, _ = data['valid']
    X, y = map(np.concatenate, ((X, X_valid), (y, y_valid)))
    del data, X_valid, y_valid

    # Training set is too bing. Sub-sample it.
    samples = np.arange(X.shape[0])
    np.random.shuffle(samples)
    samples = samples[:nb_samples_used]
    X = X[samples]
    y = y[samples]

    # Flat the features, which are 3-rank tensors
    # at the end of InceptionV3's convolutions.
    X = X.reshape(X.shape[0], -1)
    # Convert one-hot encoding to labels.
    y = np.argmax(y, -1)

    uniques, counts = np.unique(y, return_counts=True)
    print('occurrences:', dict(zip(uniques, counts)))

    model = Pipeline([
        ('pca', PCA(n_components=2024, random_state=7)),
        ('svc', SVC(class_weight='balanced', random_state=13))
    ])

    if grid_searching:
        print('grid searching...', end=' ')
        grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs)
        grid.fit(X, y)
        model = grid.best_estimator_
    else:
        print('training...', end=' ')
        model.fit(X, y)

    print('done.')

    print('saving model...', end=' ')
    joblib.dump(model, ckpt_file_name)
    print('done.')
