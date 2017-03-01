import numpy as np
import tensorflow as tf

from base import load_data, combine_pairs_for_evaluation
from model import build_network

tf.app.flags.DEFINE_string('data_dir', '/datasets/ldavid/van_gogh/vgdb_2016',
                           'directory containing the data files')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'size of the training batch')
tf.app.flags.DEFINE_integer('nb_labels', 2,
                            'number of labels occurring in the training data')
tf.app.flags.DEFINE_string('device', '/cpu:0',
                           'device in which training will occur')
tf.app.flags.DEFINE_integer('random_state', None,
                            'random State seed for replication')

FLAGS = tf.app.flags.FLAGS


def evaluate(X_pairs, y_pairs, names_pairs):
    tf.logging.info('building model...')
    t_u = tf.placeholder(tf.float32, [None, 2048], 'u_inputs')
    t_v = tf.placeholder(tf.float32, [None, 2048], 'v_inputs')

    t_fu = build_network(t_u)
    t_fv = build_network(t_v)

    distance = tf.reduce_sum((t_fu - t_fv) ** 2, axis=-1)

    optimal_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        tf.logging.info('restoring...')
        optimal_saver.restore(s, './ckpt/opt/opt.ckpt-0')

        tf.logging.info('evaluating model...')

        d = np.array(
            [s.run(distance, feed_dict={t_u: u, t_v: v})
             for (u, v), y, name in zip(X_pairs, y_pairs, names_pairs)]
        )

        print('avg distance for identities:', d[y_pairs == 1].mean())
        print('avg distance for non-identities:', d[y_pairs == 0].mean())

        T = 11

        p = d.mean(axis=-1)
        p = (p < T).astype(np.float)
        print('accuracy:', (p == y_pairs).mean())


def main(argv=None):
    print(__doc__)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    data = load_data(data_dir=FLAGS.data_dir, random_state=FLAGS.random_state)

    X_pairs, y_pairs, names_pairs = combine_pairs_for_evaluation(
        *data['train'], *data['test'],
        patches_used=40)

    with tf.device(FLAGS.device):
        evaluate(X_pairs, y_pairs, names_pairs)


if __name__ == '__main__':
    tf.app.run()
