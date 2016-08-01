"""
TensorFlow TwoLayers Model Descriptor.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, max_pool2d

from .base import Model


class TwoConvLayers(Model):
    """TensorFlow TwoLayers Model Descriptor."""

    def __init__(self, X, y=None, dropout=None, batch_size=None):
        super().__init__(X=X, y=y, dropout=dropout, batch_size=batch_size)

        with tf.variable_scope('l1'):
            y_ = convolution2d(X, 32, (3, 3))
            y_ = max_pool2d(y_, (2, 2), stride=2)

        with tf.variable_scope('l2'):
            y_ = convolution2d(y_, 64, (5, 5))
            y_ = max_pool2d(y_, (2, 2), stride=2)

        self.y_ = y_
