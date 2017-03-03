import numpy as np
from sklearn import metrics

from connoisseur.fusion import ContrastiveFusion


def combine_pairs_for_evaluation(X_base, y_base, names_base,
                                 X_eval, y_eval, names_eval,
                                 anchor_painter_label=1, patches_used=40):
    # Filter paintings by a specific artist and flat out paintings,
    # leaving only a list of patches.
    X_base = np.concatenate(X_base[y_base == anchor_painter_label])

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
            f = ContrastiveFusion(model, strategy=strategy)
            p = f.predict(X, batch_size=batch_size, threshold=threshold)
            accuracy_score = metrics.accuracy_score(y, p)

            print('score using', strategy,
                  'strategy, threshold %.1f: %.2f%%'
                  % (threshold, 100 * accuracy_score),
                  '\nConfusion matrix:\n', metrics.confusion_matrix(y, p),
                  '\nWrong predictions:', names[y != p])

            if accuracy_score > scores['test']:
                scores['test'] = accuracy_score
    return scores
