"""6 Triplet Loss Full Network Training From Scratch.

Author: Lucas David -- <lucasolivdavid@gmail.com>
License: MIT (c) 2016

"""

import os

import numpy as np
import tensorflow as tf

from base import triplets_gen_from_gen, triplet_loss
from model import build_network

from connoisseur.utils.image import ImageDataGenerator

tf.app.flags.DEFINE_string('data_dir', '/datasets/van_gogh/vgdb_2016',
                           'directory containing the data files')
tf.app.flags.DEFINE_string('logs_dir', './logs',
                           'directory to training logs.')
tf.app.flags.DEFINE_integer('anchor_label', 1,
                            'id for anchor label in triplet loss')
tf.app.flags.DEFINE_integer('batch_size', 40, 'size of the training batch')
tf.app.flags.DEFINE_integer('window_size', 40,
                            'size of window used to collect anchors, positive'
                            ' and negative samples for the triplet loss')
tf.app.flags.DEFINE_string('device', '/gpu:0',
                           'device in which training will occur')
tf.app.flags.DEFINE_integer('nb_iterations', 10000,
                            'maximum number of training iterations')
tf.app.flags.DEFINE_integer('nb_val_interval', 50,
                            'the interval between each model validation')
tf.app.flags.DEFINE_integer('nb_val_samples', 400,
                            'the number of batches used to validate')
tf.app.flags.DEFINE_integer('nb_save_interval', 1000,
                            'the interval between each session state saving')
tf.app.flags.DEFINE_integer('random_state', None,
                            'random State seed for replication')

FLAGS = tf.app.flags.FLAGS


def train():
    os.makedirs('/work/ckpt/6-triplets/training-from-scratch/opt/', exist_ok=True)
    os.makedirs('/work/ckpt/6-triplets/training-from-scratch/progress/', exist_ok=True)

    tf.logging.info('building model...')

    image_shape = [299, 299, 3]
    batch_shape = [None] + image_shape

    with tf.device(FLAGS.device):
        i_embedding = tf.placeholder(tf.float32, batch_shape, 'input_embedding')
        i_a = tf.placeholder(tf.float32, batch_shape, 'input_anchors')
        i_p = tf.placeholder(tf.float32, batch_shape, 'input_positives')
        i_n = tf.placeholder(tf.float32, batch_shape, 'input_negatives')

        f_embedding = build_network(i_embedding, is_training=True)
        f_a = build_network(i_a, is_training=True, reuse=True)
        f_p = build_network(i_p, is_training=True, reuse=True)
        f_n = build_network(i_n, is_training=True, reuse=True)
        loss_op = triplet_loss(f_a, f_p, f_n)

        tf.summary.scalar("loss", loss_op)
        merged_summary_op = tf.summary.merge_all()

        gs = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.001, gs, 20000, .5)
        opt_op = tf.train.AdamOptimizer(lr).minimize(loss_op)

    optimal_saver = tf.train.Saver(max_to_keep=1)
    progress_saver = tf.train.Saver()
    best_val_loss = np.inf

    g = ImageDataGenerator(rescale=1. / 255.)
    train_data = g.flow_from_directory(
        os.path.join(FLAGS.data_dir, 'extracted_patches', 'train'),
        target_size=image_shape[:2], augmentations=('brightness', 'contrast'),
        batch_size=FLAGS.window_size, shuffle=True, seed=None)
    valid_data = g.flow_from_directory(
        os.path.join(FLAGS.data_dir, 'extracted_patches', 'valid'),
        target_size=image_shape[:2], augmentations=('brightness', 'contrast'),
        batch_size=FLAGS.window_size, shuffle=True, seed=None)

    train_data = triplets_gen_from_gen(train_data, i_embedding, f_embedding,
                                       batch_size=FLAGS.batch_size)
    valid_data = triplets_gen_from_gen(valid_data, i_embedding, f_embedding,
                                       batch_size=FLAGS.batch_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        tf.global_variables_initializer().run()

        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, graph=tf.get_default_graph())

        tf.logging.info('training...')

        for it in range(FLAGS.nb_iterations):
            a, p, n = next(train_data)

            _, _train_loss, summary = s.run([opt_op, loss_op, merged_summary_op],
                                            feed_dict={i_a: a, i_p: p, i_n: n})

            summary_writer.add_summary(summary, it)

            if not it % FLAGS.nb_val_interval:
                print('#%i train_loss: %f' % (it, _train_loss), end=' ')

                val_loss = val_samples_seen = 0
                val_batches_seen = 0
                while val_samples_seen < FLAGS.nb_val_samples:
                    a, p, n = next(valid_data)
                    val_loss += s.run(loss_op, feed_dict={i_a: a, i_p: p, i_n: n})
                    val_batches_seen += 1
                    val_samples_seen += len(a)
                val_loss /= val_batches_seen

                print('val_loss: %f' % val_loss)

                if val_loss < best_val_loss:
                    tf.logging.info('val_acc improved from %.4f to %.4f', best_val_loss, val_loss)
                    optimal_saver.save(s, '/work/ckpt/6-triplets/training-from-scratch/opt/'
                                          'opt.ckpt', global_step=gs)
                    best_val_loss = val_loss

            if not it % FLAGS.nb_save_interval:
                progress_saver.save(s, '/work/ckpt/6-triplets/training-from-scratch/progress/'
                                       'progress.ckpt', global_step=gs)


def main(argv=None):
    print(__doc__)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    try:
        train()
    except KeyboardInterrupt:
        tf.logging.warning('interrupted by the user')
    else:
        tf.logging.info('training done.')


if __name__ == '__main__':
    tf.app.run()
