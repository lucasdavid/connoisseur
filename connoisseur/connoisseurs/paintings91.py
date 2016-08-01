import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from . import models, base


class Paintings91(base.Connoisseur):
    """Paintings91 Deep ConvNet Classifier."""

    OUTER_SCOPE = 'paintings91'

    def _build(self, X, y=None, keep_prob=tf.constant(1.0), reuse=None):
        with tf.variable_scope(self.OUTER_SCOPE, reuse=reuse):
            nn = models.AlexNET(X, y=y, dropout=keep_prob,
                                batch_size=self.batch_size)

            # Expand AlexNET with a last fc layer,
            # considering all the 91 authors.
            nn.y_ = fully_connected(nn.y_, 91, scope='fc8')

            # ... and a softmax classifier.
            nn.estimator = tf.nn.top_k(tf.nn.softmax(nn.y_))

            if y is not None:
                nn.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(nn.y_, y))
                nn.score = tf.reduce_mean(tf.to_float(
                    tf.equal(
                        tf.reshape(tf.cast(tf.argmax(y, 1), tf.int32), [-1, 1]),
                        nn.estimator[1])))
        return nn
