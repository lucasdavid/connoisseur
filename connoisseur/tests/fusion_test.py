from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from nose_parameterized.parameterized import parameterized
from sklearn.svm import LinearSVC

from connoisseur.fusion import SkLearnFusion

N_SAMPLES = 100
N_PATCHES = 5
N_CLASSES = 10


class SkLearnFusionTest(TestCase):
    @parameterized.expand([
        ('sum',),
        ('farthest',),
        ('most_frequent',),
    ])
    def test_sanity(self, strategy):
        estimator = MagicMock()
        SkLearnFusion(estimator=estimator, strategy=strategy)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            estimator = MagicMock()
            SkLearnFusion(estimator=estimator, strategy='illegal-strategy')

    @parameterized.expand([
        ('sum',),
        ('farthest',),
        ('most_frequent',),
    ])
    def test_predict(self, strategy):
        X = np.random.rand(N_SAMPLES, N_PATCHES, 14)
        y = np.random.randint(5, size=N_SAMPLES)
        estimator = LinearSVC()

        X_train = X[:10]
        X_train = X_train.reshape((-1,) + X_train.shape[2:])
        y_train = np.repeat(y[:10], N_PATCHES)

        estimator.fit(X_train, y_train)

        f = SkLearnFusion(estimator, strategy=strategy)
        p = f.predict(X)
        self.assertEqual(N_SAMPLES, p.shape[0])
        self.assertTrue(np.all(p < N_CLASSES))

    @parameterized.expand([
        ('sum',),
        ('farthest',),
        ('most_frequent',),
    ])
    def test_predict_discrimination(self, strategy):
        X = np.random.rand(N_SAMPLES, N_PATCHES, 14)
        y = np.random.randint(2, size=N_SAMPLES)
        estimator = LinearSVC()

        X_train = X[:10]
        X_train = X_train.reshape((-1,) + X_train.shape[2:])
        y_train = np.repeat(y[:10], N_PATCHES)

        estimator.fit(X_train, y_train)

        f = SkLearnFusion(estimator, strategy=strategy)
        p = f.predict(X)
        self.assertEqual(N_SAMPLES, p.shape[0])
        self.assertTrue(np.all(p < N_CLASSES))
