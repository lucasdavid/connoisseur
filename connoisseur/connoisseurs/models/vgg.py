"""TensorFlow VGG Model Descriptors.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import tensorflow as tf
from tensorflow.contrib.layers import max_pool2d, convolution2d, fully_connected

from . import base


class TruncatedVGG(base.Model):
    """TensorFlow Simple VGG Model Descriptor.

    The first eight convolutional and a fully connected layer of the
    "Very Deep Neural Network".
    """

    def __init__(self, X, y=None, dropout=None, batch_size=None):
        super().__init__(X=X, y=y)

        y_ = convolution2d(X, 64, (3, 3), scope='conv1')
        y_ = max_pool2d(y_, (2, 2), scope='pool1')

        y_ = convolution2d(y_, 64, (3, 3), scope='conv2')
        y_ = max_pool2d(y_, (2, 2), scope='pool2')

        y_ = convolution2d(y_, 128, (3, 3), scope='conv3')
        y_ = max_pool2d(y_, (2, 2), scope='pool3')

        y_ = convolution2d(y_, 128, (3, 3), scope='conv4')
        y_ = max_pool2d(y_, (2, 2), scope='pool4')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv5')
        y_ = max_pool2d(y_, (2, 2), scope='pool5')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv6')
        y_ = max_pool2d(y_, (2, 2), scope='pool6')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv7')
        y_ = max_pool2d(y_, (2, 2), scope='pool7')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv8')
        y_ = max_pool2d(y_, (2, 2), scope='pool8')

        y_ = tf.reshape(y_, (batch_size, -1))

        y_ = fully_connected(y_, 4096, scope='fc9')
        y_ = tf.nn.dropout(y_, dropout)

        y_ = fully_connected(y_, 4096, scope='fc10')
        y_ = tf.nn.dropout(y_, dropout)

        self.y_ = y_


class VGG16(base.Model):
    """TensorFlow VGG16 Model Descriptor.

    Very Deep Neural Network.
    """

    def __init__(self, X, y=None, dropout=None, batch_size=None):
        super().__init__(X=X, y=y)

        y_ = convolution2d(X, 64, (3, 3), scope='conv1')
        y_ = convolution2d(y_, 64, (3, 3), scope='conv2')
        y_ = max_pool2d(y_, (2, 2), scope='pool2')

        y_ = convolution2d(y_, 128, (3, 3), scope='conv3')
        y_ = convolution2d(y_, 128, (3, 3), scope='conv4')
        y_ = max_pool2d(y_, (2, 2), scope='pool4')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv5')
        y_ = convolution2d(y_, 256, (3, 3), scope='conv6')
        y_ = convolution2d(y_, 256, (3, 3), scope='conv7')
        y_ = max_pool2d(y_, (2, 2), scope='pool7')

        y_ = convolution2d(y_, 512, (3, 3), scope='conv8')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv9')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv10')
        y_ = max_pool2d(y_, (2, 2), scope='pool10')

        y_ = convolution2d(y_, 512, (3, 3), scope='conv11')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv12')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv13')
        y_ = max_pool2d(y_, (2, 2), scope='pool13')

        y_ = tf.reshape(y_, (batch_size, -1))

        y_ = fully_connected(y_, 4096, scope='fc14')
        y_ = tf.nn.dropout(y_, dropout)

        y_ = fully_connected(y_, 4096, scope='fc15')
        y_ = tf.nn.dropout(y_, dropout)

        self.y_ = y_


class VGG19(base.Model):
    """TensorFlow VGG19 Model Descriptor.

    Very Deep Neural Network.
    """

    def __init__(self, X, y=None, dropout=None, batch_size=None):
        super().__init__(X=X, y=y)

        y_ = convolution2d(X, 64, (3, 3), scope='conv1')
        y_ = convolution2d(y_, 64, (3, 3), scope='conv2')
        y_ = max_pool2d(y_, (2, 2), scope='pool2')

        y_ = convolution2d(y_, 128, (3, 3), scope='conv3')
        y_ = convolution2d(y_, 128, (3, 3), scope='conv4')
        y_ = max_pool2d(y_, (2, 2), scope='pool4')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv5')
        y_ = convolution2d(y_, 256, (3, 3), scope='conv6')
        y_ = max_pool2d(y_, (2, 2), scope='pool6')

        y_ = convolution2d(y_, 256, (3, 3), scope='conv7')
        y_ = convolution2d(y_, 256, (3, 3), scope='conv8')
        y_ = max_pool2d(y_, (2, 2), scope='pool8')

        y_ = convolution2d(y_, 512, (3, 3), scope='conv9')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv10')
        y_ = max_pool2d(y_, (2, 2), scope='pool10')

        y_ = convolution2d(y_, 512, (3, 3), scope='conv11')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv12')
        y_ = max_pool2d(y_, (2, 2), scope='pool12')

        y_ = convolution2d(y_, 512, (3, 3), scope='conv13')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv14')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv15')
        y_ = convolution2d(y_, 512, (3, 3), scope='conv16')
        y_ = max_pool2d(y_, (2, 2), scope='pool16')

        y_ = tf.reshape(y_, (batch_size, -1))

        y_ = fully_connected(y_, 4096, scope='fc17')
        y_ = tf.nn.dropout(y_, dropout)

        y_ = fully_connected(y_, 4096, scope='fc18')
        y_ = tf.nn.dropout(y_, dropout)

        self.y_ = y_
