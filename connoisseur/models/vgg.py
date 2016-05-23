"""
TensorFlow VGG Model Descriptor.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf

from . import base
from .utils import convolve, fully_connect, max_pool, normal_layer


class VGG(base.Model):
    """TensorFlow VGG Model Descriptor.

    Very Deep Neural Network.
    """

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        l1 = convolve(X, *normal_layer([3, 3, 3, 64]))
        l1 = tf.nn.relu(l1)

        l2 = convolve(l1, *normal_layer([3, 3, 64, 64]))
        l2 = tf.nn.relu(l2)
        l2 = max_pool(l2)

        l3 = convolve(l2, *normal_layer([3, 3, 64, 128]))
        l3 = tf.nn.relu(l3)

        l4 = convolve(l3, *normal_layer([3, 3, 128, 128]))
        l4 = tf.nn.relu(l4)
        l4 = max_pool(l4)

        l5 = convolve(l4, *normal_layer([3, 3, 128, 256]))
        l5 = tf.nn.relu(l5)

        l6 = convolve(l5, *normal_layer([3, 3, 256, 256]))
        l6 = tf.nn.relu(l6)

        l7 = convolve(l6, *normal_layer([3, 3, 256, 256]))
        l7 = tf.nn.relu(l7)

        l8 = convolve(l7, *normal_layer([3, 3, 256, 256]))
        l8 = tf.nn.relu(l8)
        l8 = max_pool(l8)

        l9 = convolve(l8, *normal_layer([3, 3, 256, 512]))
        l9 = tf.nn.relu(l9)

        l10 = convolve(l9, *normal_layer([3, 3, 512, 512]))
        l10 = tf.nn.relu(l10)

        l11 = convolve(l10, *normal_layer([3, 3, 512, 512]))
        l11 = tf.nn.relu(l11)
        l11 = max_pool(l11)

        l12 = convolve(l11, *normal_layer([3, 3, 512, 512]))
        l12 = tf.nn.relu(l12)

        l13 = convolve(l12, *normal_layer([3, 3, 512, 512]))
        l13 = tf.nn.relu(l13)

        l14 = convolve(l13, *normal_layer([3, 3, 512, 512]))
        l14 = tf.nn.relu(l14)

        l15 = convolve(l14, *normal_layer([3, 3, 512, 512]))
        l15 = tf.nn.relu(l15)
        l15 = max_pool(l15)
        l15 = tf.reshape(l15, [50, 51200])

        l16 = fully_connect(l15, *normal_layer([51200, 4096]))
        l16 = tf.nn.relu(l16)
        l16 = tf.nn.dropout(l16, dropout)

        l17 = fully_connect(l16, *normal_layer([4096, 4096]))
        l17 = tf.nn.relu(l17)
        self.last_layer = tf.nn.dropout(l17, dropout)
