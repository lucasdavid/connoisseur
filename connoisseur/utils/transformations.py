import abc

import numpy as np


class OneHot(metaclass=abc.ABCMeta):
    @classmethod
    def encode(cls, y):
        n_samples, n_classes = y.shape, np.unique(y).shape
        encoded = np.zeros((n_samples, n_classes))
        encoded[y] = 1
        return encoded

    @classmethod
    def decode(cls, y):
        return np.argmax(y, axis=1)
