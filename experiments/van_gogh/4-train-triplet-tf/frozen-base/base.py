import os
import pickle
import tensorflow as tf

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(data_dir, share_val_samples=None, random_state=None):
    data = {}
    for p in ('train', 'valid', 'test'):
        try:
            with open(os.path.join(data_dir, '%s.pickle' % p), 'rb') as f:
                data[p] = pickle.load(f)
                data[p] = (data[p]['data'],
                           data[p]['target'],
                           np.array(data[p]['names'], copy=False))
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


def triplet_loss(a, p, n):
    """Triplet Loss used in https://arxiv.org/pdf/1503.03832.pdf.
    """
    alpha = 1

    tf.cast(a > 0, tf.float32) * a

    return tf.reduce_sum(tf.nn.relu(tf.reduce_sum((a - p) ** 2, axis=-1)
                                    - tf.reduce_sum((a - n) ** 2, axis=-1)
                                    + alpha))


def triplets_gen(X, y, names, embedding_inputs, embedding_net,
                 batch_size=32, window_size=64, anchor_label=1, shuffle=True,
                 nb_epochs=np.inf, nb_samples=None, verbose=1):
    if verbose: print('%i samples belonging to %i labels' % y.shape)

    if len(y.shape) > 1:
        y = np.argmax(y, axis=-1).ravel()

    indices = np.arange(y.shape[0])
    if shuffle: np.random.shuffle(indices)

    if nb_samples is None:
        nb_samples = indices.shape[0]

    epoch = 0
    while epoch < nb_epochs:
        window_offset = 0

        while window_offset < nb_samples:
            window_indices = indices[
                             window_offset: window_offset + window_size]

            X_window, y_window = X[window_indices], y[window_indices]
            f_window = embedding_net.eval(
                feed_dict={embedding_inputs: X_window})

            p_indices = np.where(y_window == anchor_label)
            n_indices = np.where(y_window != anchor_label)

            positives, f_positives = X_window[p_indices], f_window[p_indices]
            negatives, f_negatives = X_window[n_indices], f_window[n_indices]

            # Select only hard-negatives triplets (p.4)
            triplets = np.array(
                [[a, p, negatives[
                    np.argmin(np.sum(np.abs(f_negatives - f_a), axis=-1))]]
                 for (p, f_p) in zip(positives, f_positives)
                 for a, f_a in zip(positives, f_positives)], copy=False)

            batch_offset = 0
            while batch_offset < triplets.shape[0]:
                triplets_batch = triplets[
                                 batch_offset:batch_offset + batch_size]

                yield ([triplets_batch[:, 0],
                        triplets_batch[:, 1],
                        triplets_batch[:, 2]])

                batch_offset += batch_size
            window_offset += window_size
        epoch += 1


def combine_pairs_for_evaluation(X_base, y_base, names_base,
                                 X_eval, y_eval, names_eval,
                                 anchor_label=1, patches_used=40):
    if len(y_base.shape) > 1: y_base = np.argmax(y_base, -1)
    if len(y_eval.shape) > 1: y_eval = np.argmax(y_eval, -1)

    # Filter some artist paintings (0:nvg, 1:vg).
    s = y_base == anchor_label
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

    X_pairs = [
        [
            X_base[np.random.choice(X_base.shape[0], patches_used)],
            x_eval_patches[np.random.choice(x_eval_patches.shape[0],
                                            patches_used)]
        ]
        for x_eval_patches in X_eval]
    y_pairs = (y_eval == anchor_label).astype(np.int)
    return X_pairs, y_pairs, names_eval
