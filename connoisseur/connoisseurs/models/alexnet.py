"""TensorFlow AlexNET Model Descriptor.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import tensorflow as tf
from tensorflow.contrib.layers import max_pool2d, conv2d, fully_connected

from . import base


class AlexNET(base.Model):
    """TensorFlow AlexNET Model Descriptor."""

    def __init__(self, X, y=None, dropout=None, batch_size=None):
        super().__init__(X=X, y=y, dropout=dropout, batch_size=batch_size)

        y_ = conv2d(X, 96, (11, 11), stride=4, padding='SAME', scope='conv1')
        y_ = max_pool2d(y_, 3, scope='pool1')

        y_ = conv2d(y_, 256, (5, 5), padding='SAME', scope='conv2')
        y_ = max_pool2d(y_, 3, scope='pool2')

        y_ = conv2d(y_, 384, (3, 3), padding='SAME', scope='conv3')

        y_ = conv2d(y_, 384, (3, 3), padding='SAME', scope='conv4')

        y_ = conv2d(y_, 256, (3, 3), padding='SAME', scope='conv5')
        y_ = max_pool2d(y_, 3, scope='pool5')

        y_ = tf.reshape(y_, (batch_size, -1))

        y_ = fully_connected(y_, 4096, scope='fc6')
        y_ = tf.nn.dropout(y_, dropout)

        y_ = fully_connected(y_, 4096, scope='fc7')
        self.y_ = tf.nn.dropout(y_, dropout)
