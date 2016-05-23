"""
TensorFlow Model Utils.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import tensorflow as tf
from tensorflow.python.framework import dtypes


def normal_layer(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32, seed=None, name=None):
    """Create Kernel Tensors for Convolution Layers.

    Returns a tuple (Tensor, Tensor), where the first one is the layer weights and the second one if the bias vector.
    """
    return (tf.Variable(tf.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name)),
            tf.Variable(tf.constant(0, dtype=dtypes.float32, shape=[shape[-1]])))


def convolve(X, W, b):
    conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.bias_add(conv, b)


def max_pool(X, ksize=None, strides=None, padding='SAME'):
    if not ksize: ksize = [1, 2, 2, 1]
    if not strides: strides = [1, 2, 2, 1]

    return tf.nn.max_pool(X, ksize=ksize, strides=strides, padding=padding)


def fully_connect(X, W, b):
    """Creates a fully connected layer."""
    return tf.nn.bias_add(tf.matmul(X, W), b)
