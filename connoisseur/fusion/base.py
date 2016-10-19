import abc

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from . import strategies


class Fusion(ClassifierMixin, BaseEstimator, metaclass=abc.ABCMeta):
    """Classification Fusion Base Class.

    This meta-classifier predicts data in the format
    `(n_samples, n_patches, *features)` by fusing all `n_patches` responses
    and producing a unified answer for every single one of the `n_samples`
    samples.

    Parameters
    ----------
    estimator: machine learning model
        A Keras' model or a classifier from scikit-learn.
    strategy: str, optional (default 'sum') {'sum', 'farthest', 'most_frequent'}
        sum : the probabilities for reach patch are added. Wins the class that
              occurred with most strength among all patches of a sample.
        farthest : the patch containing the highest probability (trust)
                   for one given class is selected.
        most_frequent : select the most frequent class among the patches
                        of the image.

    """

    def __init__(self, estimator, strategy='sum'):
        self.estimator = estimator
        self.strategy = strategies.get(strategy)

    def fit(self, X, y, **fit_params):
        return self.estimator.fit(X, y, **fit_params)

    def predict(self, X):
        raise NotImplementedError

    def _reduce_rank(self, probabilities, labels):
        if len(probabilities.shape) == 3 and probabilities.shape[-1] == 1:
            probabilities = probabilities.reshape(probabilities.shape[:-1])
        if len(labels.shape) == 3 and labels.shape[-1] == 1:
            labels = labels.reshape(labels.shape[:-1])

        return probabilities, labels


class SkLearnFusion(Fusion):
    def predict(self, X):
        """Predict the class for each sample in X.

        :param X: list, shaped as (paintings, patches, features).
        :return: y, predicted labels, according to a fusion strategy.
        """
        probabilities = (self.estimator
                         .decision_function(X.reshape((-1,) + X.shape[2:]))
                         .reshape(X.shape[:2] + (-1,)))
        labels = (self.estimator
                  .predict(X.reshape((-1,) + X.shape[2:]))
                  .astype(np.int)
                  .reshape(X.shape[:2] + (-1,)))

        probabilities, labels = self._reduce_rank(probabilities, labels)
        return self.strategy(labels, probabilities,
                             multi_class=len(self.estimator.classes_) > 2)


class KerasFusion(Fusion):
    def predict(self, X):
        """Predict the class for each sample in X.

        :param X: list, shaped as (paintings, patches, features).
        :return: y, predicted labels, according to a fusion strategy.
        """
        probabilities = self.estimator.predict(X.reshape((-1,) + X.shape[2:]))
        labels = np.argmax(probabilities, axis=-1)
        labels = labels.reshape(X.shape[:2] + (-1,))
        probabilities = probabilities.reshape(X.shape[:2] + (-1,))

        probabilities, labels = self._reduce_rank(probabilities, labels)
        return self.strategy(labels, probabilities, multi_class=True)
