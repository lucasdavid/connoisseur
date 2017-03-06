"""6 Evaluate Triplet.

Evaluate the 3-Dense network trained to recognize triplets of features
extracted from the InceptionV3 base in `.../2-transform-inception/run.py`
script.

Author: Lucas David -- <lucasolivdavid@gmail.com>
License: MIT (c) 2016

"""

import numpy as np
import tensorflow as tf
from sklearn import metrics
from connoisseur.fusion import strategies

from base import load_data, combine_pairs_for_evaluation
from model import build_network


tf.app.flags.DEFINE_string('data_dir', '/datasets/van_gogh/vgdb_2016',
                           'directory containing the data files')
tf.app.flags.DEFINE_string('device', '/gpu:0',
                           'device in which training will occur')
tf.app.flags.DEFINE_string('ckpt_file', '/work/ckpt/van_gogh/6-triplets/'
                                        'frozen-base/opt/'
                                        'opt.ckpt-0',
                           'checkpoint file containing the trained model')
tf.app.flags.DEFINE_integer('random_state', None,
                            'random State seed for replication')

tf.app.flags.DEFINE_float('decision_threshold', .5,
                          'Threshold used to signal max length between '
                          'the compared sample and the anchors.')

FLAGS = tf.app.flags.FLAGS


def evaluate(X, y, names):
    tf.logging.info('building model...')
    input_shape = [None, 8 * 8 * 2048]

    t_u = tf.placeholder(tf.float32, input_shape, 'u_inputs')
    t_v = tf.placeholder(tf.float32, input_shape, 'v_inputs')

    t_fu = build_network(t_u)
    t_fv = build_network(t_v, reuse=True)

    distance = tf.sqrt(tf.reduce_sum((t_fu - t_fv) ** 2, axis=-1))

    optimal_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        tf.logging.info('restoring...')
        optimal_saver.restore(s, FLAGS.ckpt_file)

        tf.logging.info('evaluating model...')

        d = np.array(
            [s.run(distance, feed_dict={t_u: u, t_v: v})
             for (u, v), y, name in zip(X, y, names)]
        )

        print('avg distance for identities:', d[y == 1].mean())
        print('avg distance for non-identities:', d[y == 0].mean())

        for threshold in (.1, .2, .3, .4, .5):
            labels = (d <= FLAGS.decision_threshold).astype(np.int)

            for strategy in ('contrastive_mean', 'most_frequent'):
                p = strategies.get(strategy)(labels, d, t=threshold)
                accuracy_score = metrics.accuracy_score(y, p)
                print('score (%s, %.1f): %.2f%%'
                      % (strategy, threshold, 100 * accuracy_score),
                      '\nConfusion matrix:\n', metrics.confusion_matrix(y, p),
                      '\nWrong predictions: %s\n' % names[y != p])


def main(argv=None):
    print(__doc__)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    data = load_data(data_dir=FLAGS.data_dir, random_state=FLAGS.random_state)

    X_train, y_train, names_train = data['train']
    X_valid, y_valid, names_valid = data['train']

    X_train, y_train, names_train = map(np.concatenate, (
        (X_train, X_valid), (y_train, y_valid), (names_train, names_valid)))

    X_pairs, y_pairs, names_pairs = combine_pairs_for_evaluation(
        X_train, y_train, y_valid, *data['test'],
        patches_used=40)

    with tf.device(FLAGS.device):
        evaluate(X_pairs, y_pairs, names_pairs)


if __name__ == '__main__':
    tf.app.run()
