"""TensorFlow VGG Model Descriptor.


Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf
from tensorflow.contrib.layers import max_pool2d, convolution2d

from . import base


class TruncatedVGG(base.Model):
    """TensorFlow Simple VGG Model Descriptor.

    The first eight convolutional and a fully connected layer of the
    "Very Deep Neural Network".
    """

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        with tf.variable_scope('l1'):
            y_ = convolution2d(X, 64, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l2'):
            y_ = convolution2d(y_, 64, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l3'):
            y_ = convolution2d(y_, 128, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l4'):
            y_ = convolution2d(y_, 128, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l5'):
            y_ = convolution2d(y_, 256, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l6'):
            y_ = convolution2d(y_, 256, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l7'):
            y_ = convolution2d(y_, 256, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        with tf.variable_scope('l8'):
            y_ = convolution2d(y_, 256, kernel_size=(3, 3))
            y_ = max_pool2d(y_, kernel_size=(2, 2))

        self.y_ = y_


class VGG(base.Model):
    """TensorFlow VGG Model Descriptor.

    Very Deep Neural Network.
    """

    def __init__(self, X, y=None, dropout=None):
        super().__init__(X=X, y=y, dropout=dropout)

        with tf.variable_scope('l1'):
            y_ = convolution2d(X, 64, (3, 3))

        with tf.variable_scope('l2'):
            y_ = convolution2d(y_, 64, (3, 3))
            y_ = max_pool2d(y_, (2, 2))

        with tf.variable_scope('l3'):
            y_ = convolution2d(y_, 128, (3, 3))

        with tf.variable_scope('l4'):
            y_ = convolution2d(y_, 128, (3, 3))
            y_ = max_pool2d(y_, (2, 2))

        with tf.variable_scope('l5'):
            y_ = convolution2d(y_, 256, (3, 3))

        with tf.variable_scope('l6'):
            y_ = convolution2d(y_, 256, (3, 3))
            y_ = max_pool2d(y_, (2, 2))

        with tf.variable_scope('l7'):
            y_ = convolution2d(y_, 256, (3, 3))

        with tf.variable_scope('l8'):
            y_ = convolution2d(y_, 256, (3, 3))
            y_ = max_pool2d(y_, (2, 2))

        with tf.variable_scope('l9'):
            y_ = convolution2d(y_, 512, (3, 3))

        with tf.variable_scope('l10'):
            y_ = convolution2d(y_, 512, (3, 3))
            y_ = max_pool2d(y_, (2, 2))

        with tf.variable_scope('l11'):
            y_ = convolution2d(y_, 512, (3, 3))

        with tf.variable_scope('l12'):
            y_ = convolution2d(y_, 512, (3, 3))
            y_ = max_pool2d(y_, (2, 2))

        with tf.variable_scope('l13'):
            y_ = convolution2d(y_, 512, (3, 3))

        with tf.variable_scope('l14'):
            y_ = convolution2d(y_, 512, (3, 3))

        with tf.variable_scope('l15'):
            y_ = convolution2d(y_, 512, (3, 3))

        with tf.variable_scope('l16'):
            y_ = convolution2d(y_, 512, (3, 3))
            self.y_ = max_pool2d(y_, (2, 2))
