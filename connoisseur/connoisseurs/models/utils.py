"""
TensorFlow Model Utils.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import tensorflow as tf


def normal_layer(shape, mean=0.0, stddev=.1, dtype=tf.float32, seed=None):
    """Create Kernel Tensors for Convolution Layers.

    Returns a tuple (Tensor, Tensor), where the first one is the layer weights
    and the second one if the bias vector.
    """
    W = tf.get_variable('weights', shape,
                        initializer=tf.random_normal_initializer(
                                         mean, stddev, seed, dtype))
    b = tf.get_variable('biases', shape[-1],
                        initializer=tf.constant_initializer(0.0, dtype))
    return W, b

