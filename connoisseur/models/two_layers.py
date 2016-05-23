"""
TensorFlow TwoLayers Model Descriptor.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf

from . import base
from .utils import convolve, max_pool, normal_layer


class TwoLayers(base.Model):
    """TensorFlow TwoLayers Model Descriptor."""

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        l1 = convolve(X, *normal_layer([5, 5, 3, 48]))
        l1 = tf.nn.relu(l1)
        l1 = tf.nn.max_pool(l1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        l2 = convolve(l1, *normal_layer([10, 10, 48, 128]))
        l2 = tf.nn.relu(l2)
        l2 = tf.nn.max_pool(l2, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        self.last_layer = tf.reshape(l2, shape=[20, 25 * 128])
        # 829440
