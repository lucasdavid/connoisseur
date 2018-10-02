import numpy as np
import tensorflow as tf
from sklearn import metrics

from base import combine_pairs_for_evaluation
from model import build_network

from connoisseur.datasets import VanGogh
from connoisseur.fusion import strategies

tf.app.flags.DEFINE_string('data_dir', '/datasets/vangogh/',
                           'directory containing the data files')
tf.app.flags.DEFINE_string('ckpt_file', '/work/ckpt/vangogh/1-triplets/'
                                        'training-from-scratch/opt/'
                                        'opt.ckpt-0',
                           'checkpoint file containing the trained model')
tf.app.flags.DEFINE_float('decision_threshold', 11.,
                          'Threshold used to signal max length between '
                          'the compared sample and the anchors.')
tf.app.flags.DEFINE_integer('dataset_seed', 48,
                            'seed for dataset random state')
tf.app.flags.DEFINE_integer('batch_size', 1024, 'size of the processing batch')
tf.app.flags.DEFINE_integer('nb_patches_compared', 40,
                            'number of patches used when comparing paintings')
tf.app.flags.DEFINE_integer('nb_patches_loaded', 40,
                            'number of patches loaded for each painting')
tf.app.flags.DEFINE_integer('nb_labels', 2,
                            'number of labels occurring in the training data')
tf.app.flags.DEFINE_integer('n_jobs', 6,
                            'number of jobs running concurrently')
tf.app.flags.DEFINE_string('device', '/gpu:0',
                           'device in which training will occur')
tf.app.flags.DEFINE_integer('random_state', None,
                            'random State seed for replication')

FLAGS = tf.app.flags.FLAGS


def evaluate():
    vangogh = VanGogh(base_dir=FLAGS.data_dir, image_shape=[299, 299, 3],
                      n_jobs=FLAGS.n_jobs,
                      train_n_patches=FLAGS.nb_patches_loaded,
                      valid_n_patches=FLAGS.nb_patches_loaded,
                      test_n_patches=FLAGS.nb_patches_loaded,
                      random_state=FLAGS.dataset_seed)
    train, valid, test = vangogh.load_patches_from_full_images()
    X_train, y_train, n_train = map(np.concatenate, zip(train, valid))
    del valid
    X_test, y_test, n_test = test
    X_train /= 255.
    X_test /= 255.
    X, y, names = combine_pairs_for_evaluation(
        X_train, y_train, n_train, X_test, y_test, n_test,
        anchor_label=1, patches_used=FLAGS.nb_patches_compared)
    del X_train, y_train, n_train, X_test, y_test, n_test, train, test

    tf.logging.info('building model...')
    image_shape = [299, 299, 3]
    batch_shape = [None] + image_shape

    t_u = tf.placeholder(tf.float32, batch_shape, 'u_inputs')
    t_v = tf.placeholder(tf.float32, batch_shape, 'v_inputs')

    t_fu = build_network(t_u)
    t_fv = build_network(t_v, reuse=True)

    distance = tf.reduce_sum((t_fu - t_fv) ** 2, axis=-1)

    optimal_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as s:
        tf.global_variables_initializer().run()

        tf.logging.info('restoring...')
        optimal_saver.restore(s, FLAGS.ckpt_file)

        tf.logging.info('evaluating model...')

        d = np.array([s.run(distance, feed_dict={t_u: u, t_v: v}) for u, v in X])

        print('avg distance for identities:', d[y == 1].mean())
        print('avg distance for non-identities:', d[y == 0].mean())

        labels = (d <= FLAGS.decision_threshold).astype(np.int)

        for threshold in (7.6e-6, 8e-6, 8.4e-6, 8.6e-6):
            for strategy in ('contrastive_mean',):
                p = strategies.get(strategy)(labels, d, t=threshold)
                accuracy_score = metrics.accuracy_score(y, p)
                print('score using', strategy,
                      'strategy, threshold %f: %.2f%%' % (threshold, 100 * accuracy_score),
                      '\nConfusion matrix:\n', metrics.confusion_matrix(y, p),
                      '\nWrong predictions: %s' % names[y != p])


def main(argv=None):
    print(__doc__)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.device(FLAGS.device):
        evaluate()


if __name__ == '__main__':
    tf.app.run()
