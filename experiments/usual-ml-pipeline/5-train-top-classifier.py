"""Train Top Classifier.

Train an SVM over the previously extracted features.

Author: Lucas David -- <lucasolivdavid@gmail.com>
Licence: MIT License 2016 (c)

"""

from sacred import Experiment

ex = Experiment('train-top-classifier')


@ex.config
def config():
    data_dir = '/datasets/vangogh/min_inception_flatten'
    ckpt_file_name = '/work/models/256/random/vgg19-%s-svm,pca:0.99.pkl'
    phases = ('train', 'valid')
    nb_samples_used = None
    chunks_loaded = [0, 1]
    grid_searching = True
    param_grid = {
        'pca__n_components': [.99],
        'svc__C': [0.1, 1.0, 10, 100, 1000],
        'svc__kernel': ['linear', 'rbf'],
        'svc__class_weight': ['balanced'],
    }
    cv = None
    n_jobs = 4
    classes = None
    patches = None
    layers = ['block%i_conv1' % i for i in range(1, 6)]


@ex.automain
def run(data_dir, phases, nb_samples_used, grid_searching, param_grid, cv, n_jobs,
        ckpt_file_name, chunks_loaded, classes, layers, patches):
    import os
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.externals import joblib
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    from connoisseur.datasets import load_pickle_data

    os.makedirs(os.path.dirname(ckpt_file_name), exist_ok=True)

    print('loading data...')
    data = load_pickle_data(data_dir=data_dir, phases=phases, chunks=chunks_loaded, layers=layers, classes=classes)
    Xs, y, _ = data['train']

    if 'valid' in data:
        # Merge train and valid data, as K-fold
        # cross-validation will be performed afterwards.
        X_valid, y_valid, _ = data['valid']

        for layer in Xs:
            Xs[layer] = np.concatenate((Xs[layer], X_valid[layer]))
        y = np.concatenate((y, y_valid))
        del X_valid, y_valid
    del data

    if nb_samples_used:
        # Training set is too bing. Sub-sample it.
        samples = np.arange(Xs.shape[0])
        np.random.shuffle(samples)
        samples = samples[:nb_samples_used]
        Xs = Xs[samples]
        y = y[samples]

    for layer, data in Xs.items():
        print('%s output shape: %s' % (layer, data.shape))
    print('y shape:', y.shape)

    uniques, counts = np.unique(y, return_counts=True)
    print('occurrences:', dict(zip(uniques, counts)))

    for layer_tag, X in Xs.items():
        print('using %s as input' % layer_tag)
        # Flat the features, which are 3-rank tensors
        # at the end of InceptionV3's convolutions.
        X = X.reshape(X.shape[0], -1)

        model = Pipeline([
            ('pca', PCA(n_components=.99, random_state=7)),
            ('svc', SVC(kernel='rbf', C=1000.0, class_weight='balanced', random_state=13, max_iter=10000000)),
        ])

        if grid_searching:
            print('grid searching...', end=' ')
            grid = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=10, refit=True)
            grid.fit(X, y)
            model = grid.best_estimator_
            print('best parameters found:', grid.best_params_)
        else:
            print('training...', end=' ')
            model.fit(X, y)

        pca = model.steps[0][1]

        print('done -- training score:', model.score(X, y),
              'pca components:', pca.n_components_,
              '(%f energy conserved)' % sum(pca.explained_variance_ratio_))
        print('saving model...', end=' ')
        joblib.dump(model, ckpt_file_name % layer_tag)
        print('done.')
