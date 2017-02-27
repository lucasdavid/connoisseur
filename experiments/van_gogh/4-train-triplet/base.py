import numpy as np
from sklearn import metrics

from connoisseur import fusion


def triplets_gen(X, y, names, embedding_net,
                 batch_size=32, window_size=64,
                 anchor_label=1, shuffle=True):
    indices = np.arange(y.shape[0])
    if shuffle: np.random.shuffle(indices)

    window_offset = 0

    while True:
        window_indices = indices[window_offset: window_offset + window_size]
        window_offset = (window_offset + window_size) % (indices.shape[0] - window_size)

        X_window, y_window = X[window_indices], y[window_indices]

        positive_indices = np.where(y_window == anchor_label)
        negative_indices = np.where(y_window != anchor_label)

        positives = X_window[positive_indices]
        negatives = X_window[negative_indices]

        # Select only hard-negatives triplets (p.4)
        hard_negatives = np.array(
            [[a, p, negatives[np.random.randint(negatives.shape[0])]]
             for p in positives
             for a in positives], copy=False)

        batch_offset = 0
        while batch_offset < hard_negatives.shape[0]:
            hard_negatives_batch = hard_negatives[batch_offset:batch_offset + batch_size]

            # X is converted to list (i.e. multiple inputs),
            # unused y is made out of dummy values.
            yield ([hard_negatives_batch[:, 0],
                    hard_negatives_batch[:, 1],
                    hard_negatives_batch[:, 2]],
                   np.repeat(-1, hard_negatives_batch.shape[0]))
            batch_offset += batch_size


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

    X_train, y_train, names_train = data['train']
    X_test, y_test, names_test = data['test']
    X, y, names = combine_pairs_for_evaluation(X_train, y_train, names_train,
                                               X_test, y_test, names_test,
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
