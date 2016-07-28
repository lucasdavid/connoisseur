import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from . import models, base


class Paintings91(base.Connoisseur):
    """Paintings91 Deep ConvNet Classifier."""

    OUTER_SCOPE = 'paintings91'

    def _build(self, X, y=None, keep_prob=tf.constant(1.0), reuse=None):
        with tf.variable_scope(self.OUTER_SCOPE, reuse=reuse):
            nn = models.VGG(X, y=y, dropout=keep_prob)
            y_ = tf.reshape(nn.y_, [-1, 2048])

            with tf.variable_scope('l17'):
                y_ = fully_connected(y_, 4096)
                y_ = tf.nn.dropout(y_, keep_prob)

            with tf.variable_scope('l18'):
                y_ = fully_connected(y_, 4096)
                y_ = tf.nn.dropout(y_, keep_prob)

            with tf.variable_scope('l19'):
                nn.y_ = fully_connected(y_, 91, activation_fn=None)

            # ... and a softmax classifier.
            nn.estimator = tf.nn.top_k(tf.nn.softmax(nn.y_))

            if y is not None:
                nn.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(nn.y_, y))
                nn.score = tf.reduce_mean(tf.to_float(
                    tf.equal(tf.reshape(tf.cast(tf.argmax(y, 1), tf.int32), [-1, 1]),
                             nn.estimator[1])))
        return nn
