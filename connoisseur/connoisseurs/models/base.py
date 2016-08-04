"""
TensorFlow Model Base Class.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""

import numpy as np
import tensorflow as tf


class Model:
    """TensorFlow Model Base Class."""

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

        self.y_ = None
        self.estimator = None
        self.loss = tf.constant(np.inf)
        self.score = tf.constant(0.0)
