from keras.engine import Model
from sklearn.base import BaseEstimator

from . import strategies
from .base import SoftMaxFusion, SkLearnFusion, ContrastiveFusion


__all__ = ['SoftMaxFusion', 'SkLearnFusion', 'ContrastiveFusion',
           'strategies']
