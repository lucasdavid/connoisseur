import numpy as np
import tensorflow as tf


def triplet_loss(a, p, n):
    """Triplet Loss used in https://arxiv.org/pdf/1503.03832.pdf.
    """
    alpha = 1

    tf.cast(a > 0, tf.float32) * a

    return tf.reduce_mean(tf.nn.relu(tf.reduce_sum((a - p) ** 2, axis=-1)
                                     - tf.reduce_sum((a - n) ** 2, axis=-1)
                                     + alpha))


def triplets_gen_from_gen(gen, embedding_inputs, embedding_net,
                          batch_size=32, anchor_label=1):
    while True:
        X_window, y_window = next(gen)
        y_window = np.argmax(y_window, axis=-1).ravel()
        f_window = embedding_net.eval(feed_dict={embedding_inputs: X_window})

        p_indices = np.where(y_window == anchor_label)
        n_indices = np.where(y_window != anchor_label)

        positives, f_positives = X_window[p_indices], f_window[p_indices]
        negatives, f_negatives = X_window[n_indices], f_window[n_indices]

        # Select only hard-negatives triplets (p.4)
        triplets = np.array(
            [[a, p, negatives[np.argmin(np.sum(np.square(f_negatives - f_a), axis=-1))]]
             for p in positives
             for a, f_a in zip(positives, f_positives)], copy=False)

        batch_offset = 0
        while batch_offset < triplets.shape[0]:
            triplets_batch = triplets[batch_offset:batch_offset + batch_size]

            yield ([triplets_batch[:, 0],
                    triplets_batch[:, 1],
                    triplets_batch[:, 2]])

            batch_offset += batch_size


def combine_pairs_for_evaluation(X_base, y_base, n_base,
                                 X_eval, y_eval, n_eval,
                                 anchor_label=1, patches_used=40):
    assert 4 <= len(X_base.shape) == len(X_eval.shape) <= 5

    if len(y_base.shape) > 1: y_base = np.argmax(y_base, -1)
    if len(y_eval.shape) > 1: y_eval = np.argmax(y_eval, -1)

    # Filter some artist paintings (0:nvg, 1:vg).
    s = y_base == anchor_label
    X_base = X_base[s]

    if len(X_base.shape) == 5:
        # Flatten patches.
        X_base = X_base.reshape(-1, *X_base.shape[2:])

    X_pairs = [[
                   X_base[np.random.choice(X_base.shape[0], patches_used)],
                   x_eval[np.random.choice(x_eval.shape[0], patches_used)]
               ] for x_eval in X_eval]
    y_pairs = (y_eval == anchor_label).astype(np.int)
    return X_pairs, y_pairs, n_eval
