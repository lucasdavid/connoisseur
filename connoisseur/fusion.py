import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin


class FusionBase(BaseEstimator, MetaEstimatorMixin):
    class StrategySet:
        @staticmethod
        def mean(s):
            return np.mean(s, axis=1)

        @staticmethod
        def sum(s):
            return np.sum(s, axis=1)

        @staticmethod
        def far(s):
            return np.argmax(s, axis=1)

    def __init__(self, estimator, strategy='far'):
        self.estimator = estimator
        self.strategy = strategy

        if not hasattr(self.StrategySet, strategy):
            raise ValueError('Illegal strategy value: %s', strategy)

    def _check(self, X, y=None):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if len(X.shape) == 3:
            # A single sample.
            X = X.reshape((1,) + X.shape)

        if len(X.shape) == 4:
            # A single patch in each sample.
            X = X.reshape((X.shape[0],) + (1,) + X.shape[1:])

        return X

    def predict(self, X):
        """Predict .

        :param X: list, shaped as (read as regular expression):
            (BATCH_SIZE?, PATCHES?, H, W, C).
        :return: y, predicted labels, according to a fusion strategy.
        """

        X = self._check(X)

        y_ = []
        strategy = getattr(self.StrategySet, self.strategy)

        for patches in X:
            _y = self.estimator.predict(patches)
            y_.append(strategy(_y))

        return np.array(y_)

    def fit(self, X, y, **fit_params):
        X, y = self._check(X, y)
        return self.estimator.fit(X, y, **fit_params)


class FusionClassifier(FusionBase, ClassifierMixin):
    def predict(self, X):
        return super().predict(X).astype(int)
