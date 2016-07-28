import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from . import models, base


class MNIST(base.Connoisseur):
    """MNIST Deep ConvNet Classifier."""

    OUTER_SCOPE = 'mnist'

    def _build(self, X, y=None, keep_prob=tf.constant(1.0), reuse=None):
        with tf.variable_scope(self.OUTER_SCOPE, reuse=reuse):
            nn = models.TwoConvLayers(X, y=y, dropout=keep_prob)

            # Extend TwoConvLayers with two fully connected layers.
            y_ = tf.reshape(nn.y_, (-1, 3136))

            with tf.variable_scope('l3'):
                y_ = fully_connected(y_, 2048)
                y_ = tf.nn.dropout(y_, keep_prob)

            with tf.variable_scope('l4'):
                nn.y_ = fully_connected(y_, 10)

            # ... and a softmax classifier.
            nn.estimator = tf.nn.top_k(tf.nn.softmax(nn.y_))

            if y is not None:
                nn.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(nn.y_, y))

                nn.score = tf.reduce_mean(tf.to_float(
                    tf.equal(tf.reshape(y, [-1, 1]),
                             nn.estimator[1])))
        return nn
