import tensorflow as tf

from . import models, base


class MNIST(base.Connoisseur):
    """MNIST Deep ConvNet Classifier."""

    OUTER_SCOPE = 'mnist'

    def _build(self, X, y=None, keep_prob=tf.constant(1.0), reuse=None):
        with tf.variable_scope(self.OUTER_SCOPE, reuse=reuse):
            nn = models.TwoConvLayers(X, y=y, dropout=keep_prob)

            y_ = tf.reshape(nn.y_, [-1, 1024])

            with tf.variable_scope('l3'):
                W, b = models.utils.normal_layer([1024, 2048])

                y_ = tf.matmul(y_, W)
                y_ = tf.nn.bias_add(y_, b)
                y_ = tf.nn.relu(y_)
                y_ = tf.nn.dropout(y_, keep_prob)

            with tf.variable_scope('l4'):
                W, b = models.utils.normal_layer([2048, 10])

                nn.y_ = tf.nn.bias_add(tf.matmul(y_, W), b)

            # ... and a softmax classifier.
            nn.estimator = tf.nn.top_k(tf.nn.softmax(nn.y_))

            if y is not None:
                nn.loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(nn.y_, y))

                nn.score = tf.reduce_mean(tf.to_float(
                    tf.equal(tf.reshape(y, [-1, 1]),
                             nn.estimator[1])))
        return nn
