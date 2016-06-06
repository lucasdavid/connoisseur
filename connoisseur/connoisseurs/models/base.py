"""
TensorFlow Model Base Class.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import abc

import numpy as np
import tensorflow as tf


class Model(metaclass=abc.ABCMeta):
    """TensorFlow Model Base Class."""

    def __init__(self, X, y=None, dropout=None):
        self.X = X
        self.y = y
        self.dropout = dropout

        self.y_ = None
        self.estimator = None
        self.loss = tf.constant(np.inf)
        self.score = tf.constant(0.0)
