"""Train Top Classifier.

Train an SVM over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

import os

import joblib
import numpy as np
from sacred import Experiment
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from connoisseur.datasets import load_pickle_data, group_by_paintings

ex = Experiment('train-top-classifier')


@ex.config
def config():
    data_dir = '/mnt/files/datasets/vgdb_2016/embed/patches/random/'
    ckpt_file_name = 'model.pkl'
    phases = ('train','valid')
    nb_samples_used = None
    chunks_loaded = [0]
    grid_searching = False
    param_grid = {
        # 'pca__n_components': [.95, .99],
        'svc__C': [0.1, 1.0, 10, 100, 1000],
        'svc__kernel': ['rbf'],
        'svc__class_weight': [None, 'balanced'],
    }
    cv = None
    n_jobs = 4
    classes = None
    max_patches = 50
    layer = 'global_average_pooling2d'
    using_pca = True


@ex.automain
def run(_run, data_dir, phases, nb_samples_used, grid_searching, param_grid, cv, n_jobs,
        ckpt_file_name, chunks_loaded, classes, layer, max_patches, using_pca):
    report_dir = _run.observers[0].dir

    print('loading data...')
    data = load_pickle_data(data_dir=data_dir, phases=phases,
                            chunks=chunks_loaded,
                            layers=[layer], classes=classes)
    x, y, names = data['train']
    x = x[layer]

    if 'valid' in data:
        # Merge train and valid data, as K-fold
        # cross-validation will be performed afterwards.
        x_valid, y_valid, names_valid = data['valid']
        x_valid = x_valid[layer]

        x = np.concatenate((x, x_valid))
        y = np.concatenate((y, y_valid))
        names = np.concatenate((names, names_valid))
        del x_valid, y_valid, names_valid
    del data

    x, y, names = group_by_paintings(x, y, names=names, max_patches=max_patches)
    x, y = map(np.concatenate, (x, y))

    if nb_samples_used:
        # Training set is too bing. Sub-sample it.
        samples = np.arange(x.shape[0])
        np.random.shuffle(samples)
        samples = samples[:nb_samples_used]
        x = x[samples]
        y = y[samples]

    print('%s output shape: %s' % (layer, x.shape))
    print('occurrences:', dict(zip(*np.unique(y, return_counts=True))))

    # Flat the features, which are 3-rank tensors
    # at the end of InceptionV3's convolutions.
    x = x.reshape(x.shape[0], -1)

    steps = []
    steps.append(('stan', StandardScaler()))

    if using_pca:
        steps.append(('pca', PCA(n_components=.99, random_state=7)))

    steps.append(('svc', SVC(max_iter=10000)))
    model = Pipeline(steps)

    if grid_searching:
        print('grid searching...', end=' ')
        grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=10, refit=True)
        grid.fit(x, y)
        model = grid.best_estimator_
        print('best parameters found:', grid.best_params_)
    else:
        print('training...', end=' ')
        model.fit(x, y)

    if using_pca:
        try:
            pca = model.steps[1][1]
            print('done -- training score:', model.score(x, y),
                'pca components:', pca.n_components_,
                '(%f energy conserved)' % sum(pca.explained_variance_ratio_))
        except: ...

    print('saving model...', end=' ')
    joblib.dump(model, os.path.join(report_dir, ckpt_file_name))
    print('done.')
