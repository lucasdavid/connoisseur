import math

import numpy as np
from sklearn.metrics import classification_report

from connoisseur import fusion


def distance_to_label(p, threshold=.5):
    return (p.ravel() < threshold).astype(np.float)


def combine_pairs_for_training_gen(X, y, names, batch_size):
    samples0, = np.where(y == 0)
    samples1, = np.where(y == 1)

    while True:
        c0 = np.hstack((
            np.random.choice(samples1, size=(math.floor(batch_size / 2), 1)),
            np.random.choice(samples0, size=(math.floor(batch_size / 2), 1))))
        c1 = np.random.choice(samples1, size=(math.ceil(batch_size / 2), 2))
        c = np.vstack((c0, c1))

        X_pairs = X[c.T]

        y_pairs = np.concatenate((np.zeros(math.floor(batch_size / 2)),
                                  np.ones(math.ceil(batch_size / 2))))

        yield list(X_pairs), y_pairs


def combine_pairs_for_evaluation(data, batch_size):
    for phase in ('train', 'test'):
        # Aggregate patches by their respective paintings.
        X, y, names = [], [], []
        X_all, y_all, names_all = data[phase]
        # Remove patches indices, leaving just the painting name.
        names_all = np.array(['-'.join(n.split('-')[:-1]) for n in names_all],
                             copy=False)
        for name in set(names_all):
            s = names_all == name
            X.append(X_all[s])
            y.append(y_all[s][0])
            names.append(names_all[s][0])
        X, y, names = map(np.array, (X, y, names))

        data[phase] = X, y, names

    X, y, names = data['train']
    X_test, y_test, names_test = data['test']

    # Aggregate train samples by their respective painter.
    _X = []
    for label in np.unique(y):
        s = y == label
        _X.append((X[s], y[s], names[s]))
    X = np.array(_X)

    Z = []
    for i in range(X_test.shape[0]):
        for j in range(X_test[i].shape[0]):
            pass

    for x, label, name in zip(X_test, y_test, names_test):
        for x_train, label_train, name_train in zip(X, y, names):
            pass


    yield 1


def evaluate(model, data, batch_size):
    print('evaluating model with patch fusion strategies')

    pairs_flow = combine_pairs_for_evaluation(data, batch_size=batch_size)

    y, p = [], []
    for batch_i in range(20000):
        X, _y = next(pairs_flow)

        # fusion.SkLearnFusion()

        _p = distance_to_label(model.predict_on_batch(X))
        y.append(_y)
        p.append(_p)

    y, p = np.concatenate(y), np.concatenate(p)
    print('\n# %s' % phase)
    print(classification_report(y, p))
    print('accuracy:', (p == y).mean())

    del X, y, p
