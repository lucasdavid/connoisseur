"""Dense Network Example.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (c) 2016

"""

import os

import numpy as np
import tensorflow as tf

from base import load_data, triplets_gen, triplet_loss
from model import build_network

tf.app.flags.DEFINE_string('data_dir', '/datasets/ldavid/van_gogh/vgdb_2016',
                           'directory containing the data files')
tf.app.flags.DEFINE_integer('anchor_label', 1,
                            'id for anchor label in triplet loss')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'size of the training batch')
tf.app.flags.DEFINE_integer('window_size', 512,
                            'size of window used to collect anchors, positive'
                            ' and negative samples for the triplet loss')
tf.app.flags.DEFINE_string('device', '/gpu:0',
                           'device in which training will occur')
tf.app.flags.DEFINE_integer('nb_iterations', 10000,
                            'maximum number of training iterations')
tf.app.flags.DEFINE_integer('nb_val_interval', 50,
                            'the interval between each model validation')
tf.app.flags.DEFINE_float('share_val_samples', .1,
                          'share of samples used for validation')
tf.app.flags.DEFINE_integer('nb_save_interval', 1000,
                            'the interval between each session state saving')
tf.app.flags.DEFINE_integer('random_state', None,
                            'random State seed for replication')

FLAGS = tf.app.flags.FLAGS


def train(X_train, y_train, names_train,
          X_valid, y_valid, names_valid):
    os.makedirs('./ckpt/opt/', exist_ok=True)
    os.makedirs('./ckpt/progress/', exist_ok=True)

    tf.logging.info('building model...')

    with tf.device(FLAGS.device):
        i_embedding = tf.placeholder(tf.float32, [None, 2048],
                                     'input_embedding')
        i_a = tf.placeholder(tf.float32, [None, 2048], 'input_anchors')
        i_p = tf.placeholder(tf.float32, [None, 2048], 'input_positives')
        i_n = tf.placeholder(tf.float32, [None, 2048], 'input_negatives')

        f_embedding = build_network(i_embedding)
        f_a = build_network(i_a, is_training=True)
        f_p = build_network(i_p, is_training=True)
        f_n = build_network(i_n, is_training=True)
        loss = triplet_loss(f_a, f_p, f_n)

        gs = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.001, gs, 250, .5)
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

    optimal_saver = tf.train.Saver(max_to_keep=1)
    progress_saver = tf.train.Saver()
    best_val_loss = np.inf

    train_data = triplets_gen(X_train, y_train, names_train,
                              i_embedding, f_embedding,
                              batch_size=FLAGS.batch_size,
                              window_size=FLAGS.window_size)
    valid_data = triplets_gen(X_valid, y_valid, names_valid,
                              i_embedding, f_embedding,
                              batch_size=FLAGS.batch_size,
                              window_size=FLAGS.window_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        tf.global_variables_initializer().run()

        tf.logging.info('training...')

        for it in range(FLAGS.nb_iterations):
            a, p, n = next(train_data)

            _, _train_loss = s.run([optimizer, loss],
                                   feed_dict={i_a: a, i_p: p, i_n: n})

            if not it % FLAGS.nb_val_interval:
                print('#%i train_loss: %f' % (it, _train_loss), end=' ')

                a, p, n = next(valid_data)
                _val_loss = s.run(loss, feed_dict={i_a: a, i_p: p, i_n: n})
                print('val_loss: %f' % _val_loss)

                if _val_loss < best_val_loss:
                    tf.logging.info('val_acc improved from %.4f to %.4f',
                                    best_val_loss, _val_loss)
                    optimal_saver.save(s, './ckpt/opt/opt.ckpt',
                                       global_step=gs)
                    best_val_loss = _val_loss

            if not it % FLAGS.nb_save_interval:
                progress_saver.save(s, './ckpt/progress/progress.ckpt',
                                    global_step=gs)


def main(argv=None):
    print(__doc__)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    data = load_data(data_dir=FLAGS.data_dir,
                     phases=('train',),
                     share_val_samples=FLAGS.share_val_samples,
                     random_state=FLAGS.random_state)
    X_train, y_train, names_train = data['train']
    X_valid, y_valid, names_valid = data['valid']

    try:
        train(X_train, y_train, names_train, X_valid, y_valid, names_valid)
    except KeyboardInterrupt:
        tf.logging.warning('interrupted by the user')
    else:
        tf.logging.info('training done.')


if __name__ == '__main__':
    tf.app.run()
