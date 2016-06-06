"""TensorFlow VGG Model Descriptor.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf

from . import base
from .utils import normal_layer


def conv_layer(X, W, b):
    y_ = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    y_ = tf.nn.bias_add(y_, b)
    return tf.nn.relu(y_)


def max_pool(X):
    return tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


class TruncatedVGG(base.Model):
    """TensorFlow Simple VGG Model Descriptor.

    The first eight convolutional and a fully connected layer of the
    "Very Deep Neural Network".
    """

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        with tf.device('/gpu:1'):
            with tf.variable_scope('l1'):
                y_ = max_pool(conv_layer(X, *normal_layer([3, 3, 3, 64])))

            with tf.variable_scope('l2'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 64, 64])))

            with tf.variable_scope('l3'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 64, 128])))

            with tf.variable_scope('l4'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 128, 128])))

            with tf.variable_scope('l5'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 128, 256])))

            with tf.variable_scope('l6'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 256, 256])))

            with tf.variable_scope('l7'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 256, 256])))

            with tf.variable_scope('l8'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 256, 256])))

        self.y_ = y_


class VGG(base.Model):
    """TensorFlow VGG Model Descriptor.

    Very Deep Neural Network.
    """

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        with tf.device('/gpu:1'):
            with tf.variable_scope('l1'):
                y_ = conv_layer(X, *normal_layer([3, 3, 3, 64]))

            with tf.variable_scope('l2'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 64, 64])))

            with tf.variable_scope('l3'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 64, 128]))

            with tf.variable_scope('l4'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 128, 128])))

            with tf.variable_scope('l5'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 128, 256]))

            with tf.variable_scope('l6'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 256, 256]))
                y_ = max_pool(y_)

            with tf.variable_scope('l7'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 256, 256]))

            with tf.variable_scope('l8'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 256, 256])))

            with tf.variable_scope('l9'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 256, 512]))

            with tf.variable_scope('l10'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 512, 512]))
                y_ = max_pool(y_)

            with tf.variable_scope('l11'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 512, 512]))

            with tf.variable_scope('l12'):
                y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 512, 512])))

            with tf.variable_scope('l13'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 512, 512]))

            with tf.variable_scope('l14'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 512, 512]))
                y_ = max_pool(y_)

            with tf.variable_scope('l15'):
                y_ = conv_layer(y_, *normal_layer([3, 3, 512, 512]))

            with tf.variable_scope('l16'):
                self.y_ = max_pool(conv_layer(y_, *normal_layer([3, 3, 512, 512])))
