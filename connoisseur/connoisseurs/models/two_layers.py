"""
TensorFlow TwoLayers Model Descriptor.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf

from . import base
from .utils import normal_layer


class TwoConvLayers(base.Model):
    """TensorFlow TwoLayers Model Descriptor."""

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        with tf.variable_scope('l1'):
            W, b = normal_layer([5, 5, 3, 32])

        y_ = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
        y_ = tf.nn.bias_add(y_, b)
        y_ = tf.nn.relu(y_)
        y_ = tf.nn.max_pool(y_, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                            padding='SAME')

        with tf.variable_scope('l2'):
            W, b = normal_layer([7, 7, 32, 64])

        y_ = tf.nn.conv2d(y_, W, strides=[1, 1, 1, 1], padding='SAME')
        y_ = tf.nn.bias_add(y_, b)
        y_ = tf.nn.relu(y_)
        y_ = tf.nn.max_pool(y_, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1],
                            padding='SAME')

        self.y_ = tf.reshape(y_, shape=[-1, 25 * 64])
