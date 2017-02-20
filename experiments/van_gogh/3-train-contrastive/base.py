import math

import numpy as np
from sklearn import metrics

from connoisseur import fusion


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


def combine_pairs_for_evaluation(X_base, y_base, names_base,
                                 X_eval, y_eval, names_eval,
                                 anchor_painter_label=1, patches_used=40):
    # Filter some artist paintings (0:nvg, 1:vg).
    s = y_base == anchor_painter_label
    X_base, y_base = X_base[s], y_base[s]

    # Aggregate test patches by their respective paintings.
    _X, _y, _names = [], [], []
    # Remove patches indices, leaving just the painting name.
    names = np.array(['-'.join(n.split('-')[:-1]) for n in names_eval])
    for name in set(names):
        s = names == name
        _X.append(X_eval[s])
        _y.append(y_eval[s][0])
        _names.append(names[s][0])
    X_eval, y_eval, names_eval = map(np.array, (_X, _y, _names))

    X_pairs = [[x_eval_patches[
                    np.random.choice(x_eval_patches.shape[0], patches_used)],
                X_base[np.random.choice(X_base.shape[0], patches_used)]]
               for x_eval_patches in X_eval]
    y_pairs = (y_eval == anchor_painter_label).astype(np.int)
    return X_pairs, y_pairs, names_eval


def evaluate(model, data, batch_size):
    print('evaluating model with patch fusion strategies')

    X, y, names = combine_pairs_for_evaluation(*data['train'], *data['test'],
                                               anchor_painter_label=1,
                                               patches_used=40)
    scores = {'test': -1}
    for threshold in (.2, .3, .5):
        for strategy in ('contrastive_avg', 'most_frequent'):
            p = fusion.ContrastiveFusion(model, strategy=strategy,
                                         threshold=threshold).predict(X)
            accuracy_score = metrics.accuracy_score(y, p)
            print('score using', strategy,
                  'strategy, threshold %.1f: %.2f%%'
                  % (threshold, 100 * accuracy_score),
                  '\nConfusion matrix:\n', metrics.confusion_matrix(y, p),
                  '\nWrong predictions: %s' % names[y != p])

            if accuracy_score > scores['test']:
                scores['test'] = accuracy_score
    return scores
