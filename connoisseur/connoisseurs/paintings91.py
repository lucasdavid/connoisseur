import tensorflow as tf

from . import models, base


class Paintings91(base.Connoisseur):
    """Paintings91 Deep ConvNet Classifier."""

    OUTER_SCOPE = 'paintings91'

    def _build(self, X, y=None, keep_prob=tf.constant(1.0), reuse=None):
        with tf.variable_scope(self.OUTER_SCOPE, reuse=reuse):
            nn = models.VGG(X, y=y, dropout=keep_prob)
            y_ = tf.reshape(nn.y_, [-1, 2048])

            with tf.variable_scope('l17'):
                W, b = models.utils.normal_layer([2048, 4096])

            y_ = tf.matmul(y_, W)
            y_ = tf.nn.bias_add(y_, b)
            y_ = tf.nn.relu(y_)
            y_ = tf.nn.dropout(y_, keep_prob)

            with tf.variable_scope('l18'):
                W, b = models.utils.normal_layer([4096, 91])

            nn.y_ = tf.nn.bias_add(tf.matmul(y_, W), b)

            # ... and a softmax classifier.
            nn.estimator = tf.nn.top_k(tf.nn.softmax(nn.y_))

            if y is not None:
                nn.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(nn.y_, y))
                nn.score = tf.reduce_mean(tf.to_float(
                    tf.equal(tf.reshape(tf.cast(tf.argmax(y, 1), tf.int32), [-1, 1]),
                             nn.estimator[1])))
        return nn
