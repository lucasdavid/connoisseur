import numpy as np
from sklearn.base import ClassifierMixin

from . import strategies


class Fusion(ClassifierMixin):
    """Classification Fusion Base Class.

    This meta-classifier predicts data in the format
    `(n_samples, n_patches, *features)` by fusing all `n_patches` responses
    and producing a unified answer for every single one of the `n_samples`
    samples.

    Parameters
    ----------
    strategy: a function defined in connoisseur.fusion.strategies
    """

    def __init__(self, strategy=strategies.sum, multi_class=True):
        self.strategy = strategy
        self.multi_class = multi_class

    def predict(self, probabilities=None, hyperplane_distance=None, labels=None):
        """Predict the class for each sample.

        :param probabilities: array-like, shaped as (paintings, patches, labels).
        :param hyperplane_distance: array-like, shaped as (paintings, patches, 1).
        :param labels: array-like, shaped as (paintings, patches).
        :return: y, predicted labels, according to a fusion strategy.
        """
        assert probabilities is not None or all(x is not None for x in (labels, hyperplane_distance))

        if labels is None:
            labels = np.argmax(probabilities, axis=-1)

        if probabilities is None:
            # The same operations are applied to these two measures. We can
            # therefore join them and simplify the equations ahead.
            probabilities = hyperplane_distance

        return self.strategy(labels, probabilities, multi_class=self.multi_class)


class ContrastiveFusion(Fusion):
    def __init__(self, estimator, strategy='sum'):
        super().__init__(strategy=strategy, multi_class=False)
        self.estimator = estimator
        assert self.strategy in (strategies.contrastive_mean,
                                 strategies.most_frequent), \
            ('ContrastiveFusion only accept contrastive_mean and '
             'most_frequent strategies')

    def predict(self, X, threshold=.5, batch_size=32):
        probabilities = np.array([self.estimator.predict(
            x, batch_size=batch_size) for x in X], copy=False)
        labels = self.distance_to_label(probabilities, threshold=threshold)
        probabilities, labels = self._reduce_rank(probabilities, labels)

        return self.strategy(labels, probabilities,
                             t=threshold,
                             multi_class=False)

    def distance_to_label(self, p, threshold=.5):
        return (p < threshold).astype(np.int)
